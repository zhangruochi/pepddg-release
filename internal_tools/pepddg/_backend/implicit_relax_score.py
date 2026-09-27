"""
Implicit Relax + Score (OpenMM)

This module provides a fast, GPU-friendly implicit-solvent binding energy proxy
that is useful as an intermediate "gate" between docking and expensive explicit
solvent MD.

Core metric:
    dG_bind = E_complex - E_receptor - E_ligand

Where each energy is the OpenMM potential energy (amber14 + OBC2 implicit) and
the receptor/ligand energies are evaluated at the *minimized complex* coordinates
(no further minimization).

The minimization protocol defaults to a robust two-stage approach that applies
positional restraints to backbone heavy atoms, allowing side chains to relax while
reducing large-scale drift:
    1) strong backbone restraints
    2) weaker backbone restraints

This logic was extracted and generalized from:
    research/skempi_ddg_openmm_benchmark/skempi_ddg_openmm.py
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

KJ_MOL_TO_KCAL_MOL = 0.2390057361376673
_ONE_4PI_EPS0 = 138.935456  # kJ/mol*nm/e^2

# "Screened" electrostatics for xint energies.
# We keep the full LJ term but damp Coulomb at all distances using a high dielectric
# (water-like, ~80) so xint_screened is less dominated by charge artifacts.
# Soft-core distance used to damp close-contact Coulomb artifacts when computing the
# "screened" cross-interaction term. A slightly larger value improves ranking stability
# on peptide/cyclic peptide benchmarks without materially changing the LJ component.
_DEFAULT_COULOMB_SCREENING_DISTANCE_NM = 0.10
_DEFAULT_COULOMB_SCREENING_DIELECTRIC = 80.0


def _safe_chain_id(chain: str) -> str:
    chain_id = (chain or "").strip()
    return chain_id if chain_id else "A"


# ============================================================================
# R2 Round 2 (2026-05-12) — defect fix helpers
#
# Three fixes ported in from benchmark-side and explicit-solvent paths so
# pose_score works on:
#   (1) multi-chain receptors           — _consolidate_receptor_chains_in_pdb
#   (2) OpenMM NoneType _finalize bug   — handled inline in _minimize_and_score
#   (3) head-to-tail amide cyclics      — _locate_head_to_tail_atom_indices +
#                                          _post_add_amide_bond_to_system +
#                                          _is_same_covalent_bond
# ============================================================================

# Standard amide bond parameters (AMBER14SB-compatible, matches the values
# used in cyclic_peptide_solvate.py for the explicit-solvent path).
_AMIDE_R0_NM = 0.1335  # 1.335 Å equilibrium C-N bond length
_AMIDE_K_KJ_MOL_NM2 = 410000.0  # ~98 kcal/mol/Å² in kJ/mol/nm²


def _strip_altloc_residues(pdb_path: Any) -> None:
    """In-place: collapse alt-loc variants to the highest-occupancy conformer.

    Rationale: PDBFixer preserves alt-loc A/B atoms (e.g. CYS 165 in 7K2F /
    7K2I with occupancies 0.32 / 0.68). OpenMM amber14 template matching
    then fails with "set of atoms matches CYS, but the bonds are different"
    because two CA / CB / SG atoms exist for the same residue with
    conflicting connectivity.

    For each (chain, resnum, icode, atom_name) group, keep the line with
    the highest occupancy (ties → first occurrence) and clear column 17
    (alt-loc) to a space. Non-ATOM / non-HETATM lines pass through.
    """
    from pathlib import Path as _Path

    path = _Path(str(pdb_path))
    if not path.exists():
        return
    text = path.read_text()
    lines = text.splitlines()

    # Pass 1: group ATOM/HETATM lines by (chain, resnum, icode, atom_name)
    # and resolve the winner (highest occupancy).
    keep_idx: set[int] = set()
    groups: dict[tuple[str, str, str, str], list[tuple[int, float]]] = {}
    for idx, line in enumerate(lines):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 60:
            keep_idx.add(idx)
            continue
        altloc = line[16]
        if altloc == " ":
            keep_idx.add(idx)
            continue
        atom_name = line[12:16].strip()
        chain = line[21]
        resnum = line[22:26].strip()
        icode = line[26] if len(line) > 26 else " "
        try:
            occ = float(line[54:60].strip() or "0")
        except ValueError:
            occ = 0.0
        groups.setdefault((chain, resnum, icode, atom_name), []).append((idx, occ))

    for _, candidates in groups.items():
        # Sort by occupancy desc, then by file order asc to break ties.
        winner_idx = sorted(candidates, key=lambda t: (-t[1], t[0]))[0][0]
        keep_idx.add(winner_idx)

    # Pass 2: emit kept lines, clearing the alt-loc column.
    out_lines: list[str] = []
    for idx, line in enumerate(lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if idx not in keep_idx:
                continue
            if len(line) > 16:
                line = line[:16] + " " + line[17:]
        out_lines.append(line)

    # Preserve trailing newline behavior.
    new_text = "\n".join(out_lines)
    if text.endswith("\n"):
        new_text += "\n"
    path.write_text(new_text)


def _consolidate_receptor_chains_in_pdb(
    pdb_path: Any,
    receptor_chain: str,
    ligand_chain: str,
) -> None:
    """In-place: if the PDB at `pdb_path` has multiple chain IDs that should
    all be considered receptor (per `receptor_chain` API contract: chain ID
    pool minus the ligand_chain), consolidate them into a single chain
    `receptor_chain` with SEQUENTIAL renumbering across original chains +
    TER records between original chain boundaries.

    Rationale (defect 1): when a source CIF/PDB has multiple receptor chains
    (e.g. 5TH2 with A+C+B+D as antigen + 2 antibody chains), PDBFixer keeps
    the original chain IDs. If the caller passes receptor_chain="A" and
    there are also residues in chains "C", "D" the user *meant* as receptor,
    those residues are NOT picked up by the receptor/ligand split. Worse:
    if upstream pipeline pre-merged them into chain A without TER records,
    OpenMM AMBER14's template matcher sees fake peptide bonds between
    original chain ends → "CALA/CGLU/CASN bonds differ" errors.

    Implementation: pure-Python PDB text manipulation (no gemmi/openmm dep);
    safe to call after PDBFixer.addMissingAtoms() and before downstream
    add_cyclic_conect_records.

    Algorithm:
      Pass 1: detect chain transitions inside receptor_chain (consecutive
              ATOM lines in chain `receptor_chain` where residue numbering
              decreases or jumps backward → boundary).
      Pass 2: rewrite ATOM lines with consecutive resnum + insert TER after
              each detected boundary's last atom.

    No-op when receptor_chain has only one contiguous segment.
    """
    from pathlib import Path as _Path

    path = _Path(str(pdb_path))
    if not path.exists():
        return
    text = path.read_text()
    lines = text.splitlines()

    # First pass: identify atoms in receptor_chain + their original resnum.
    # Detect boundaries: residue number drops to a smaller value compared
    # to the prior residue in the SAME chain (indicates chain re-numbering
    # restart from a different source chain).
    receptor_atom_lines: list[tuple[int, str]] = []  # (line_idx, line)
    boundaries: list[int] = []  # line_idx where boundary occurs (this line starts new segment)
    prev_resnum: int | None = None
    prev_icode: str | None = None
    for idx, line in enumerate(lines):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 26 or line[21] != receptor_chain:
            continue
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26] if len(line) > 26 else " "
        if prev_resnum is not None and resnum < prev_resnum:
            boundaries.append(idx)
        receptor_atom_lines.append((idx, line))
        prev_resnum = resnum
        prev_icode = icode

    if not boundaries:
        return  # single segment — no consolidation needed

    # Build new residue mapping: (orig_resnum, icode, segment_id) -> new_resnum
    res_map: dict[tuple, int] = {}
    next_new_resnum = 0
    segment_id = 0
    for i, (idx, line) in enumerate(receptor_atom_lines):
        # Detect segment transition by checking against boundaries
        if idx in boundaries:
            segment_id += 1
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26] if len(line) > 26 else " "
        key = (segment_id, resnum, icode)
        if key not in res_map:
            next_new_resnum += 1
            res_map[key] = next_new_resnum

    # Pass 2: rewrite lines + insert TER records.
    out_lines: list[str] = []
    atom_serial = 0
    prev_emitted_segment_id: int | None = None
    last_receptor_atom_line: str | None = None
    current_segment_id = 0
    receptor_indices = {idx for idx, _ in receptor_atom_lines}
    for idx, line in enumerate(lines):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            # Skip original TER records inside receptor segments (we re-emit ours);
            # keep TER records outside receptor (e.g., between receptor and ligand).
            if line.startswith("TER"):
                # If this TER follows our last receptor atom, emit a fresh one and continue
                if last_receptor_atom_line is not None and prev_emitted_segment_id == current_segment_id:
                    atom_serial += 1
                    ter = (
                        "TER   "
                        + f"{atom_serial:5d}"
                        + "      "
                        + last_receptor_atom_line[17:27]
                        + " " * 53
                    )[:80]
                    out_lines.append(ter)
                    last_receptor_atom_line = None
                continue
            out_lines.append(line)
            continue

        if idx in receptor_indices:
            if idx in boundaries:
                # Emit TER for previous segment before starting new one
                if last_receptor_atom_line is not None:
                    atom_serial += 1
                    ter = (
                        "TER   "
                        + f"{atom_serial:5d}"
                        + "      "
                        + last_receptor_atom_line[17:27]
                        + " " * 53
                    )[:80]
                    out_lines.append(ter)
                current_segment_id += 1
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                out_lines.append(line)
                continue
            icode = line[26] if len(line) > 26 else " "
            new_resnum = res_map[(current_segment_id, resnum, icode)]
            atom_serial += 1
            new_line = (
                line[:6]
                + f"{atom_serial:5d}"
                + line[11:21]
                + receptor_chain
                + f"{new_resnum:4d}"
                + " "
                + (line[27:] if len(line) > 27 else "")
            )
            last_receptor_atom_line = new_line
            prev_emitted_segment_id = current_segment_id
            out_lines.append(new_line)
        else:
            # Before emitting non-receptor atom, flush TER for last receptor segment if pending
            if last_receptor_atom_line is not None and line[21] != receptor_chain:
                atom_serial += 1
                ter = (
                    "TER   "
                    + f"{atom_serial:5d}"
                    + "      "
                    + last_receptor_atom_line[17:27]
                    + " " * 53
                )[:80]
                out_lines.append(ter)
                last_receptor_atom_line = None
            atom_serial += 1
            new_line = line[:6] + f"{atom_serial:5d}" + line[11:]
            out_lines.append(new_line)

    # If receptor was the last chain, emit final TER
    if last_receptor_atom_line is not None:
        atom_serial += 1
        ter = (
            "TER   "
            + f"{atom_serial:5d}"
            + "      "
            + last_receptor_atom_line[17:27]
            + " " * 53
        )[:80]
        out_lines.append(ter)

    path.write_text("\n".join(out_lines) + "\n")


def _is_same_covalent_bond(b1: Any, b2: Any) -> bool:
    """Compare two CovalentBond objects (atom1_chain, atom1_residue, atom1_atom,
    atom2_chain, atom2_residue, atom2_atom). Used to filter head_to_tail from
    ligand_covalent_bonds list."""
    if b1 is None or b2 is None:
        return False
    try:
        return (
            getattr(b1, "atom1_chain", None) == getattr(b2, "atom1_chain", None)
            and getattr(b1, "atom1_residue", None) == getattr(b2, "atom1_residue", None)
            and getattr(b1, "atom1_atom", None) == getattr(b2, "atom1_atom", None)
            and getattr(b1, "atom2_chain", None) == getattr(b2, "atom2_chain", None)
            and getattr(b1, "atom2_residue", None) == getattr(b2, "atom2_residue", None)
            and getattr(b1, "atom2_atom", None) == getattr(b2, "atom2_atom", None)
        )
    except Exception:
        return False


def _locate_head_to_tail_atom_indices(
    topology: Any,
    ligand_chain: str,
) -> tuple[int | None, int | None]:
    """Locate the N atom of residue 1 and the C atom of residue N (last) in
    the ligand chain. Returns (n1_atom_index, cN_atom_index) — both global
    topology atom indices, or (None, None) if either cannot be found.

    Used by R2-T04 head_to_tail HarmonicBondForce injection.
    """
    n1_idx: int | None = None
    cN_idx: int | None = None
    ligand_residues: list[Any] = []
    for chain in topology.chains():
        if chain.id == ligand_chain:
            ligand_residues = list(chain.residues())
            break
    if not ligand_residues:
        return None, None

    first_res = ligand_residues[0]
    last_res = ligand_residues[-1]

    for atom in first_res.atoms():
        if atom.name == "N":
            n1_idx = atom.index
            break
    for atom in last_res.atoms():
        if atom.name == "C":
            cN_idx = atom.index
            break
    return n1_idx, cN_idx


def _post_add_amide_bond_to_system(
    system: Any,
    n_atom_index: int,
    c_atom_index: int,
    r0_nm: float = _AMIDE_R0_NM,
    k_kj_mol_nm2: float = _AMIDE_K_KJ_MOL_NM2,
) -> None:
    """Add a HarmonicBondForce term between N(res 1) and C(res N) atoms to
    represent the head-to-tail amide bond. Re-uses existing HarmonicBondForce
    if one is present; otherwise adds a new one.

    Mirror of cyclic_peptide_solvate._post_add_amide_bond, adapted for the
    implicit-solvent path. Local openmm import per module convention (the
    backend lazy-imports openmm at first call to avoid heavy load at module
    import time).
    """
    import openmm as _omm  # local lazy import
    bond_force = None
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, _omm.HarmonicBondForce):
            bond_force = f
            break
    if bond_force is None:
        bond_force = _omm.HarmonicBondForce()
        system.addForce(bond_force)
    bond_force.addBond(int(c_atom_index), int(n_atom_index), r0_nm, k_kj_mol_nm2)


def _is_nonetype_finalize_error(exc: BaseException) -> bool:
    """R2-T03 (defect 3): identify the OpenMM C++ binding cleanup bug where
    LocalEnergyMinimizer raises:
        AttributeError: 'NoneType' object has no attribute '_finalize'
    This is a transient bug in OpenMM's Context cleanup path that occasionally
    fires on specific topologies (notably cross-docked cyclic peptides).
    Returning True triggers the per-restart NaN fallback in _minimize_and_score.
    """
    if not isinstance(exc, AttributeError):
        return False
    msg = str(exc)
    return "_finalize" in msg and "NoneType" in msg


@dataclass(frozen=True)
class ChainSplitPaths:
    minimized_complex_pdb: str
    receptor_pdb: str
    ligand_pdb: str


_CHAIN_RESNUM_RE = re.compile(r"^([A-Za-z])(\d+)$")


def _parse_chain_resnum(token: str) -> Optional[Tuple[str, int]]:
    """
    Parse residue tokens like "A36" or "A:36" -> ("A", 36).
    """
    t = str(token or "").strip()
    if not t:
        return None
    if ":" in t:
        chain_s, res_s = t.split(":", 1)
        chain = _safe_chain_id(chain_s.strip())
        try:
            return chain, int(str(res_s).strip())
        except Exception:
            return None
    m = _CHAIN_RESNUM_RE.match(t)
    if not m:
        return None
    chain = _safe_chain_id(m.group(1))
    try:
        resnum = int(m.group(2))
    except Exception:
        return None
    return chain, resnum


def _residue_matches(residue: Any, resnum: int) -> bool:
    """
    Match an OpenMM Topology residue against a PDB-style residue number.

    Topology residue.id is usually the PDB resseq as a string, but some pipelines
    rely on residue.index. This helper keeps selection robust.
    """
    try:
        if str(residue.id).strip() == str(int(resnum)):
            return True
    except Exception:
        pass
    try:
        if int(residue.id) == int(resnum):
            return True
    except Exception:
        pass
    try:
        return (int(residue.index) + 1) == int(resnum)
    except Exception:
        return False


def _platform_candidates(requested: str) -> List[str]:
    req = (requested or "CPU").strip()
    req_norm = req.upper()
    if req_norm == "CUDA":
        return ["CUDA", "OpenCL", "CPU"]
    if req_norm == "OPENCL":
        return ["OpenCL", "CPU"]
    if req_norm == "CPU":
        return ["CPU"]
    return [req, "CPU"]


def _properties_for_platform(
    platform_name: str,
    *,
    device_index: int,
    cpu_threads: int,
    precision: str,
) -> Dict[str, str]:
    name = (platform_name or "").strip().upper()
    props: Dict[str, str] = {}
    if name in {"CUDA", "OPENCL"}:
        props["DeviceIndex"] = str(int(device_index))
        props["Precision"] = str(precision or "mixed").strip()
    elif name == "CPU":
        props["Threads"] = str(max(1, int(cpu_threads)))
    return props


def _split_complex_by_chain(
    pdb_path: str,
    receptor_chain: str,
    ligand_chain: str,
    out_dir: Path,
) -> ChainSplitPaths:
    receptor_chain = _safe_chain_id(receptor_chain)
    ligand_chain = _safe_chain_id(ligand_chain)

    minimized_complex = out_dir / "complex_minimized.pdb"
    receptor_raw = out_dir / "receptor_minimized.raw.pdb"
    ligand_raw = out_dir / "ligand_minimized.raw.pdb"

    # Split by simple PDB record filtering (ATOM/HETATM). Keep CONECT/SSBOND via re-add step.
    # NOTE: this assumes docking poses write receptor+ligand in separate chains.
    with open(pdb_path, "r") as f_in, open(receptor_raw, "w") as f_rec, open(ligand_raw, "w") as f_lig:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")):
                chain = _safe_chain_id(line[21].strip())
                if chain == receptor_chain:
                    f_rec.write(line)
                elif chain == ligand_chain:
                    f_lig.write(line)
            elif line.startswith("TER"):
                # Keep TER for both to preserve chain termination; write it if it refers to chain.
                # TER line has chain in column 22 in most PDBs.
                chain = _safe_chain_id(line[21].strip())
                if chain == receptor_chain:
                    f_rec.write(line)
                elif chain == ligand_chain:
                    f_lig.write(line)
            elif line.startswith(("CONECT", "SSBOND")):
                # Ignore here; we'll regenerate via cyclic_peptide_utils.
                continue
            else:
                # Keep minimal footer lines (END) in both.
                if line.startswith("END"):
                    f_rec.write(line)
                    f_lig.write(line)

    # Write minimized complex copy
    minimized_complex.write_text(Path(pdb_path).read_text())

    return ChainSplitPaths(
        minimized_complex_pdb=str(minimized_complex),
        receptor_pdb=str(receptor_raw),
        ligand_pdb=str(ligand_raw),
    )


def compute_binding_energy_implicit(
    pdb_path: str,
    receptor_chain: str,
    ligand_chain: str,
    platform: str = "CUDA",
    device_index: int = 0,
    cpu_threads: int = 1,
    *,
    work_dir: Optional[str] = None,
    fix_structure: bool = True,
    minimize_max_iterations: int = 10,
    min_protocol: str = "twostage",
    stage1_iters: int = 200,
    stage2_iters: int = 300,
    restraint_k1: float = 1000.0,
    restraint_k2: float = 100.0,
    n_restarts: int = 3,
    jitter_nm: float = 0.02,
    aggregate: str = "median",
    random_seed: int = 0,
    precision: str = "mixed",
    cross_interaction_interface_cutoff_a: float = 0.0,
    restraint_exclusion_residues: Optional[Sequence[str]] = None,
    restraint_exclusion_distance_a: float = 0.0,
    coulomb_screening_distance_nm: float = _DEFAULT_COULOMB_SCREENING_DISTANCE_NM,
    coulomb_screening_dielectric: float = _DEFAULT_COULOMB_SCREENING_DIELECTRIC,
    ligand_base_jitter_backbone_nm: float = 0.0,
    ligand_base_jitter_sidechain_nm: float = 0.0,
    ligand_base_jitter_local_radius_a: float = 0.0,
    local_backbone_restart_jitter_nm: float = 0.0,
    implicit_solvent: str = "obc2",
    force_field: str = "amber14-all.xml",
    solute_dielectric: float = 1.0,
    linker: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fast implicit-solvent binding energy proxy:
        dG_bind = E_complex - E_receptor - E_ligand

    Notes:
    - The complex is minimized (HBonds constraints, OBC2 implicit, NoCutoff).
    - Receptor-only and ligand-only energies are computed at the minimized coordinates
      (no additional minimization).
    """
    try:
        import openmm  # type: ignore
        import openmm.app as app  # type: ignore
        import openmm.unit as unit  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("OpenMM is required for compute_binding_energy_implicit()") from e

    from .cyclic_peptide_utils import (
        add_cyclic_conect_records,
        detect_cyclic_peptide,
        detect_disulfide_bonds,
    )
    from pdbfixer import PDBFixer  # type: ignore

    receptor_chain = _safe_chain_id(receptor_chain)
    ligand_chain = _safe_chain_id(ligand_chain)

    input_path = Path(pdb_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    min_protocol_norm = (min_protocol or "legacy").strip().lower()
    if min_protocol_norm not in {"legacy", "twostage"}:
        raise ValueError(f"Unsupported min_protocol: {min_protocol!r} (expected 'legacy' or 'twostage')")

    aggregate_norm = (aggregate or "median").strip().lower()
    if aggregate_norm not in {"median", "mean", "best"}:
        raise ValueError(f"Unsupported aggregate: {aggregate!r} (expected 'median', 'mean', or 'best')")

    ligand_base_jitter_backbone_nm = max(0.0, float(ligand_base_jitter_backbone_nm))
    ligand_base_jitter_sidechain_nm = max(0.0, float(ligand_base_jitter_sidechain_nm))
    ligand_base_jitter_local_radius_a = max(0.0, float(ligand_base_jitter_local_radius_a))
    local_backbone_restart_jitter_nm = max(0.0, float(local_backbone_restart_jitter_nm))

    requested = (platform or "CPU").strip()
    requested_norm = requested.upper()

    stage1_iters = max(0, int(stage1_iters))
    stage2_iters = max(0, int(stage2_iters))
    minimize_max_iterations = max(0, int(minimize_max_iterations))
    n_restarts = max(1, int(n_restarts))
    jitter_nm = max(0.0, float(jitter_nm))
    random_seed = int(random_seed)
    precision_norm = (precision or "mixed").strip()
    cross_interaction_interface_cutoff_a = float(max(0.0, float(cross_interaction_interface_cutoff_a)))
    restraint_exclusion_distance_a = float(max(0.0, float(restraint_exclusion_distance_a)))
    coulomb_screening_distance_nm = float(max(0.0, float(coulomb_screening_distance_nm)))
    coulomb_screening_dielectric = float(max(1.0, float(coulomb_screening_dielectric)))

    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="implicit_relax_")  # noqa: SIM115
        work_path = Path(tmp_ctx.name)
    else:
        tmp_ctx = None
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

    try:
        # TBMB crosslinker: strip ligand + HG before PDBFixer (which can't handle TBMB).
        # Restraints are applied to the System after createSystem() below.
        tbmb_restraints: List[Any] = []
        linker_norm = (linker or "").strip().lower()
        if linker_norm == "tbmb":
            from .tbmb_md_support import detect_tbmb_crosslinks, prepare_tbmb_for_md

            tbmb_info = detect_tbmb_crosslinks(str(input_path))
            if tbmb_info is not None and tbmb_info.is_valid:
                stripped_pdb, tbmb_info, tbmb_restraints = prepare_tbmb_for_md(
                    str(input_path),
                    str(work_path),
                    peptide_chain=ligand_chain,
                )
                input_path = Path(stripped_pdb)
                print(
                    f"TBMB: stripped ligand, {len(tbmb_restraints)} SG-SG restraints "
                    f"(CYS: {[f'{cl.cys_chain}{cl.cys_resnum}' for cl in tbmb_info.crosslinks]})"
                )
            else:
                raise ValueError(
                    f"linker=tbmb but TBMB not detected in {pdb_path}. "
                    "Expected: 9 carbon HETATM atoms (LIG) with 3 CONECT to CYS SG."
                )

        # R2-T05 (defect 5): sanitize alt-loc conformers BEFORE any disulfide
        # / cyclization detection. Raw crystal PDBs (e.g. 7K2F, 7K2I) carry
        # CYS sidechains with alt-loc A/B at close SG–SG distance (2.25 Å).
        # detect_disulfide_bonds() then mis-classifies the two altlocs as a
        # self-disulfide on the same residue (CYS165↔CYS165), which adds a
        # spurious CONECT 1251 1251 and breaks amber14 template matching.
        # Always sanitize into work_path (idempotent helper, negligible cost);
        # never skip even when caller pre-staged the PDB into work_path.
        sanitized_input = work_path / f"{input_path.stem}__altloc_sanitized.pdb"
        sanitized_input.write_text(input_path.read_text())
        _strip_altloc_residues(sanitized_input)
        input_path = sanitized_input

        # Docking poses often lack hydrogens and can be missing atoms. Fixing here makes the
        # gate robust and consistent across engines.
        fixed_input = input_path
        pre_bonds: Sequence[Any] = detect_disulfide_bonds(str(input_path))

        # Preserve non-disulfide cyclization (head-to-tail / side-chain) for the ligand chain.
        # This is critical for cyclic peptides because OpenMM topology loses CONECT on writes.
        ligand_covalent_bonds: List[Any] = []
        try:
            cyclic_info = detect_cyclic_peptide(str(input_path), peptide_chain=ligand_chain)
            ligand_covalent_bonds = list(getattr(cyclic_info, "covalent_bonds", []) or [])
        except Exception:
            ligand_covalent_bonds = []
        if fix_structure:
            fixer = PDBFixer(filename=str(input_path))
            # Snapshot the requested partition chains BEFORE removeHeterogens so we
            # can detect (and clearly report) the case where it deletes an entire
            # interface partner.
            _chains_before = {str(ch.id) for ch in fixer.topology.chains()}
            fixer.removeHeterogens(keepWater=False)
            # removeHeterogens strips ALL non-standard residues (HETATM heterogens).
            # A grafted ncAA ligand written as `HETATM ... LIG` is a heterogen, so
            # this call can delete the ENTIRE ligand (or receptor) chain. The
            # downstream chain-split then writes an atom-less, END-only PDB, and
            # `app.PDBFile()` of that file raises the cryptic
            #   AttributeError: 'NoneType' object has no attribute '_finalize'
            # from openmm/app/internal/pdbstructure.py (an END before any atom/MODEL,
            # so `_current_model is None`). That error is DETERMINISTIC for a given
            # input (not the intermittent C++ minimizer-cleanup bug the per-restart
            # `_is_nonetype_finalize_error` catchers assume), so retries never help.
            # Surface the real root cause here instead of letting it masquerade as a
            # minimization failure. This forces an HONEST abstain in callers (e.g. the
            # ncaa_interface_ddg scorer), whose amber14-only force field cannot
            # parameterize a non-standard ncAA ligand anyway — that path must route a
            # grafted ncAA through a GAFF-capable scorer, not this amber14 one.
            _chains_after = {str(ch.id) for ch in fixer.topology.chains()}
            for _part_label, _part_chain in (("receptor", receptor_chain), ("ligand", ligand_chain)):
                _pc = _safe_chain_id(_part_chain)
                if _pc in _chains_before and _pc not in _chains_after:
                    raise RuntimeError(
                        f"removeHeterogens deleted the entire {_part_label} chain "
                        f"{_pc!r} (its residues are HETATM heterogens, e.g. a grafted "
                        f"ncAA written as 'HETATM ... LIG'). amber14-all.xml cannot "
                        f"parameterize a non-standard ncAA ligand; route this complex "
                        f"through a GAFF-capable scorer instead of "
                        f"compute_binding_energy_implicit()."
                    )
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            fixed_raw = work_path / "fixed_heavy.raw.pdb"
            with open(fixed_raw, "w") as f:
                app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

            # R2-T02 (defect 1): multi-chain receptor handling. If receptor was
            # multi-chain in source, PDBFixer may have produced a single chain
            # without TER records → OpenMM AMBER14 sees fake peptide bonds
            # between original chain ends → "CALA/CGLU/CASN bonds differ"
            # template error. Consolidate: sequential renumber + TER insertion.
            _consolidate_receptor_chains_in_pdb(
                pdb_path=fixed_raw,
                receptor_chain=receptor_chain,
                ligand_chain=ligand_chain,
            )

            # R2-T05 (defect 5): collapse PDBFixer alt-loc conformers to the
            # highest-occupancy variant per atom. Without this, e.g. CYS 165
            # in 7K2F/7K2I (alt-loc A occ 0.32 + B occ 0.68) yields amber14
            # "set of atoms matches CYS, but the bonds are different".
            _strip_altloc_residues(fixed_raw)

            post_bonds = detect_disulfide_bonds(str(fixed_raw))

            # R2-T04 (defect 4): head-to-tail amide handling. If a head-to-tail
            # closure is present, strip it from the CONECT-record list so OpenMM
            # does NOT see the N1→CN amide at template-matching time (which would
            # cause "No template found" errors). The bond is restored as a
            # HarmonicBondForce AFTER createSystem (see post-createSystem hook).
            head_to_tail_present = False
            try:
                _ht_info = detect_cyclic_peptide(str(fixed_raw), peptide_chain=ligand_chain)
                if getattr(_ht_info, "head_to_tail_bond", None) is not None:
                    head_to_tail_present = True
            except Exception:
                head_to_tail_present = False
            if head_to_tail_present:
                ht_bond = _ht_info.head_to_tail_bond
                filtered_ligand_covalent_bonds = [
                    b for b in ligand_covalent_bonds
                    if not _is_same_covalent_bond(b, ht_bond)
                ]
            else:
                filtered_ligand_covalent_bonds = ligand_covalent_bonds

            fixed_input = work_path / "fixed_heavy.pdb"
            add_cyclic_conect_records(
                str(fixed_raw),
                str(fixed_input),
                disulfide_bonds=(post_bonds or pre_bonds),
                covalent_bonds=filtered_ligand_covalent_bonds,
            )

        pdb = app.PDBFile(str(fixed_input))
        _SOLVENT_MAP = {"obc2": "implicit/obc2.xml", "gbn2": "implicit/gbn2.xml", "obc1": "implicit/obc1.xml"}
        solvent_xml = _SOLVENT_MAP.get(implicit_solvent, f"implicit/{implicit_solvent}.xml")
        forcefield = app.ForceField(force_field, solvent_xml)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield, pH=7.0)

        # Persist the "fixed" input used for scoring (useful for debugging/repro).
        fixed_out_raw = work_path / "fixed.raw.pdb"
        with open(fixed_out_raw, "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)
        fixed_out = work_path / "fixed.pdb"
        fixed_bonds = detect_disulfide_bonds(str(fixed_out_raw))
        # R2-T04 (defect 4): same head_to_tail filter as above. The fixed.pdb is
        # for posterity; keep filtered to remain consistent with the topology
        # actually fed to createSystem below.
        add_cyclic_conect_records(
            str(fixed_out_raw),
            str(fixed_out),
            disulfide_bonds=(fixed_bonds or pre_bonds),
            covalent_bonds=filtered_ligand_covalent_bonds if fix_structure else ligand_covalent_bonds,
        )

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        # R2-T04 (defect 4): if head_to_tail closure was present, restore the
        # N1→CN amide as a HarmonicBondForce term (template-matching free).
        # Standard amide params: r0=1.335 Å, k=410000 kJ/mol/nm² (~98 kcal/mol/Å²).
        # See third-party/molecular_dynamics/cyclic_peptide_solvate.py::_post_add_amide_bond
        # for the equivalent explicit-solvent path.
        _head_to_tail_added_to_complex = False
        if fix_structure and head_to_tail_present:
            n1_idx, cN_idx = _locate_head_to_tail_atom_indices(
                modeller.topology, ligand_chain
            )
            if n1_idx is not None and cN_idx is not None:
                _post_add_amide_bond_to_system(system, n1_idx, cN_idx)
                _head_to_tail_added_to_complex = True
            else:
                # Could not locate N1/CN — log; treated as unsupported.
                # Per audit WARN #3, we mark this case as unsupported rather
                # than silently degrading scoring quality.
                raise RuntimeError(
                    "head_to_tail closure detected but N1/CN atom indices "
                    f"could not be located in modeller.topology for chain "
                    f"{ligand_chain!r}. head_to_tail amide cyclic peptide "
                    "is not supported by the current AMBER14 template path."
                )

        # Optionally override the solute dielectric in the GB force.
        # OpenMM 8.x encodes OBC2 as CustomGBForce with soluteDielectric
        # hardcoded in energy expressions (not as a global parameter).
        if solute_dielectric != 1.0:
            gb_forces = [
                (i, system.getForce(i))
                for i in range(system.getNumForces())
                if isinstance(system.getForce(i), openmm.CustomGBForce)
            ]
            if len(gb_forces) == 1:
                _, gb = gb_forces[0]
                new_val = f"soluteDielectric={solute_dielectric}"
                modified = 0
                for j in range(gb.getNumEnergyTerms()):
                    expr, typ = gb.getEnergyTermParameters(j)
                    if "soluteDielectric=1" in expr:
                        expr = expr.replace("soluteDielectric=1", new_val)
                        gb.setEnergyTermParameters(j, expr, typ)
                        modified += 1
                if modified == 0:
                    raise RuntimeError(
                        "No energy terms with soluteDielectric=1 found in "
                        f"CustomGBForce. Cannot set solute_dielectric={solute_dielectric}."
                    )
            elif len(gb_forces) == 0:
                import warnings
                warnings.warn(
                    f"solute_dielectric={solute_dielectric} requested but no "
                    f"CustomGBForce found in the system. Ignoring.",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    f"Expected 0 or 1 CustomGBForce, found {len(gb_forces)}. "
                    f"Cannot set solute_dielectric={solute_dielectric}."
                )

        # Apply TBMB flat-bottom SG-SG distance restraints to the complex System.
        # Use dedicated force group (10) so restraint energy can be excluded from dG_bind.
        _TBMB_FORCE_GROUP = 10
        if tbmb_restraints:
            from .tbmb_md_support import apply_flat_bottom_restraints

            n_forces_before = system.getNumForces()
            n_tbmb = apply_flat_bottom_restraints(
                system, modeller.topology, modeller.positions, tbmb_restraints
            )
            # Assign the newly added TBMB force to a separate group
            if n_tbmb > 0 and system.getNumForces() > n_forces_before:
                system.getForce(system.getNumForces() - 1).setForceGroup(_TBMB_FORCE_GROUP)
            print(f"Applied {n_tbmb} TBMB flat-bottom restraints (force group {_TBMB_FORCE_GROUP})")

        BACKBONE = {"N", "CA", "C", "O", "OXT"}
        backbone_atom_indices: List[int] = []
        sidechain_heavy_indices: List[int] = []
        ligand_backbone_heavy_indices: List[int] = []
        ligand_sidechain_heavy_indices: List[int] = []
        for chain in modeller.topology.chains():
            chain_id = _safe_chain_id(getattr(chain, "id", ""))
            for res in chain.residues():
                for atom in res.atoms():
                    element = atom.element.symbol.upper() if atom.element is not None else ""
                    if element == "H":
                        continue
                    atom_name = (atom.name or "").strip().upper()
                    idx = int(atom.index)
                    if atom_name in BACKBONE:
                        backbone_atom_indices.append(idx)
                        if chain_id == ligand_chain:
                            ligand_backbone_heavy_indices.append(idx)
                    else:
                        sidechain_heavy_indices.append(idx)
                        if chain_id == ligand_chain:
                            ligand_sidechain_heavy_indices.append(idx)

        base_pos_nm = np.array(
            [[float(p.x), float(p.y), float(p.z)] for p in modeller.positions.value_in_unit(unit.nanometer)],
            dtype=float,
        )

        ref_atom_indices: List[int] = []
        excluded_backbone: set[int] = set()
        local_backbone_indices: List[int] = []
        if restraint_exclusion_residues and restraint_exclusion_distance_a > 0.0 and backbone_atom_indices:
            parsed: List[Tuple[str, int]] = []
            for tok in restraint_exclusion_residues:
                pr = _parse_chain_resnum(str(tok))
                if pr is not None:
                    parsed.append(pr)

            if parsed:
                for chain in modeller.topology.chains():
                    chain_id = _safe_chain_id(getattr(chain, "id", ""))
                    for res in chain.residues():
                        for c, r in parsed:
                            if chain_id == _safe_chain_id(c) and _residue_matches(res, int(r)):
                                for a in res.atoms():
                                    ref_atom_indices.append(int(a.index))

            if ref_atom_indices:
                ref_coords = base_pos_nm[np.asarray(ref_atom_indices, dtype=int), :]
                bb_coords = base_pos_nm[np.asarray(backbone_atom_indices, dtype=int), :]
                diffs = bb_coords[:, None, :] - ref_coords[None, :, :]
                d2 = np.sum(diffs * diffs, axis=-1)
                min_d2 = np.min(d2, axis=1)
                thr2 = (float(restraint_exclusion_distance_a) / 10.0) ** 2
                bad = np.where(min_d2 < thr2)[0].tolist()
                for bi in bad:
                    excluded_backbone.add(int(backbone_atom_indices[int(bi)]))
                    local_backbone_indices.append(int(backbone_atom_indices[int(bi)]))

        # Localize jitter to the same neighborhood used for restraint exclusion. This keeps
        # multi-start sampling focused on the mutation site instead of injecting noise into
        # remote side chains (important for ranking/ΔΔG stability).
        jitter_atom_indices: List[int] = list(sidechain_heavy_indices)
        if restraint_exclusion_residues and restraint_exclusion_distance_a > 0.0 and sidechain_heavy_indices and ref_atom_indices:
            ref_coords = base_pos_nm[np.asarray(ref_atom_indices, dtype=int), :]
            sc_indices = np.asarray(sidechain_heavy_indices, dtype=int)
            sc_coords = base_pos_nm[sc_indices, :]
            diffs = sc_coords[:, None, :] - ref_coords[None, :, :]
            d2 = np.sum(diffs * diffs, axis=-1)
            min_d2 = np.min(d2, axis=1)
            thr2 = (float(restraint_exclusion_distance_a) / 10.0) ** 2
            keep = np.where(min_d2 < thr2)[0].tolist()
            jitter_atom_indices = [int(sc_indices[int(i)]) for i in keep]

        # Optional ligand-local "base jitter" applied once (shared across restarts). This is
        # intended as a cheap conformer ensemble knob for peptide/macrocycle ligands while
        # keeping the complex pose mostly fixed.
        ligand_base_jitter_indices: List[int] = []
        if (ligand_base_jitter_backbone_nm > 0.0 or ligand_base_jitter_sidechain_nm > 0.0) and (
            ligand_backbone_heavy_indices or ligand_sidechain_heavy_indices
        ):
            ligand_base_jitter_indices = list(ligand_backbone_heavy_indices) + list(ligand_sidechain_heavy_indices)
            if ligand_base_jitter_local_radius_a > 0.0 and ref_atom_indices and ligand_base_jitter_indices:
                ref_coords = base_pos_nm[np.asarray(ref_atom_indices, dtype=int), :]
                lig_idx = np.asarray(ligand_base_jitter_indices, dtype=int)
                lig_coords = base_pos_nm[lig_idx, :]
                diffs = lig_coords[:, None, :] - ref_coords[None, :, :]
                d2 = np.sum(diffs * diffs, axis=-1)
                min_d2 = np.min(d2, axis=1)
                thr2 = (float(ligand_base_jitter_local_radius_a) / 10.0) ** 2
                keep = np.where(min_d2 < thr2)[0].tolist()
                ligand_base_jitter_indices = [int(lig_idx[int(i)]) for i in keep]

        base_pos_nm_jittered = np.array(base_pos_nm, copy=True)
        if ligand_base_jitter_indices:
            rng0 = np.random.default_rng(int(random_seed))
            bb_set = set(int(i) for i in ligand_backbone_heavy_indices)
            sc_set = set(int(i) for i in ligand_sidechain_heavy_indices)
            for atom_idx in ligand_base_jitter_indices:
                sigma = 0.0
                if int(atom_idx) in bb_set:
                    sigma = float(ligand_base_jitter_backbone_nm)
                elif int(atom_idx) in sc_set:
                    sigma = float(ligand_base_jitter_sidechain_nm)
                if sigma <= 0.0:
                    continue
                delta = rng0.normal(loc=0.0, scale=sigma, size=(3,)).astype(float)
                max_norm = float(sigma) * 3.0
                if max_norm > 0.0:
                    nrm = float(np.linalg.norm(delta))
                    if nrm > max_norm:
                        delta = delta * (max_norm / nrm)
                base_pos_nm_jittered[int(atom_idx), :] = base_pos_nm_jittered[int(atom_idx), :] + delta

        restraint_force = None
        if min_protocol_norm == "twostage":
            restraint_force = openmm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
            restraint_force.addGlobalParameter("k", 0.0)
            restraint_force.addPerParticleParameter("x0")
            restraint_force.addPerParticleParameter("y0")
            restraint_force.addPerParticleParameter("z0")

            for idx in backbone_atom_indices:
                if int(idx) in excluded_backbone:
                    continue
                p = base_pos_nm[idx]
                restraint_force.addParticle(idx, [float(p[0]), float(p[1]), float(p[2])])
            system.addForce(restraint_force)

        platform_obj = None
        properties: Dict[str, str] = {}
        simulation: Optional[app.Simulation] = None
        last_error: Optional[Exception] = None
        for cand in _platform_candidates(requested):
            try:
                plat = openmm.Platform.getPlatformByName(cand)
            except Exception as e:
                last_error = e
                continue
            props = _properties_for_platform(
                cand,
                device_index=device_index,
                cpu_threads=cpu_threads,
                precision=precision_norm,
            )
            # Prefer deterministic CUDA kernels when available to reduce run-to-run noise
            # (important for ranking stability on small peptide/macrocycle benchmarks).
            if str(cand or "").strip().upper() == "CUDA":
                try:
                    prop_names = {str(x) for x in plat.getPropertyNames()}
                except Exception:
                    prop_names = set()
                if "DeterministicForces" in prop_names:
                    props.setdefault("DeterministicForces", "true")
            integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            try:
                simulation = app.Simulation(modeller.topology, system, integ, plat, props)
            except Exception as e:
                last_error = e
                continue
            platform_obj = plat
            properties = props
            requested = cand
            break

        if simulation is None or platform_obj is None:
            raise RuntimeError(f"Failed to initialize OpenMM Simulation (platform={platform}): {last_error}")

        def _minimize_and_score(restart_idx: int) -> Dict[str, Any]:
            restart_dir = work_path / f"restart_{restart_idx:02d}"
            restart_dir.mkdir(parents=True, exist_ok=True)

            pos_nm = np.array(base_pos_nm_jittered, copy=True)
            if restart_idx > 0 and jitter_nm > 0 and jitter_atom_indices:
                rng = np.random.default_rng(int(random_seed) + int(restart_idx))
                noise = rng.normal(loc=0.0, scale=float(jitter_nm), size=(len(jitter_atom_indices), 3)).astype(float)
                max_norm = float(jitter_nm) * 3.0
                if max_norm > 0.0:
                    norms = np.linalg.norm(noise, axis=1)
                    scale = np.ones_like(norms)
                    too_big = norms > max_norm
                    scale[too_big] = max_norm / norms[too_big]
                    noise = noise * scale[:, None]
                for local_i, atom_idx in enumerate(jitter_atom_indices):
                    pos_nm[int(atom_idx), :] = pos_nm[int(atom_idx), :] + noise[int(local_i), :]
            if restart_idx > 0 and local_backbone_restart_jitter_nm > 0 and local_backbone_indices:
                rng_bb = np.random.default_rng(int(random_seed) + 1000003 + int(restart_idx))
                noise_bb = rng_bb.normal(
                    loc=0.0,
                    scale=float(local_backbone_restart_jitter_nm),
                    size=(len(local_backbone_indices), 3),
                ).astype(float)
                max_norm_bb = float(local_backbone_restart_jitter_nm) * 3.0
                if max_norm_bb > 0.0:
                    norms = np.linalg.norm(noise_bb, axis=1)
                    scale = np.ones_like(norms)
                    too_big = norms > max_norm_bb
                    scale[too_big] = max_norm_bb / norms[too_big]
                    noise_bb = noise_bb * scale[:, None]
                for local_i, atom_idx in enumerate(local_backbone_indices):
                    pos_nm[int(atom_idx), :] = pos_nm[int(atom_idx), :] + noise_bb[int(local_i), :]
            simulation.context.setPositions(pos_nm * unit.nanometer)

            # R2-T03 (defect 3): catch the OpenMM '_finalize' AttributeError that
            # occasionally fires inside LocalEnergyMinimizer cleanup. Re-raise as
            # RuntimeError so the outer restart loop captures it cleanly with
            # error_msg referencing _finalize (rather than a cryptic AttributeError).
            try:
                if min_protocol_norm == "legacy":
                    openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=minimize_max_iterations)
                else:
                    if restraint_force is None:
                        raise RuntimeError("Internal error: restraint_force missing for twostage protocol")
                    simulation.context.setParameter("k", float(restraint_k1))
                    if stage1_iters > 0:
                        openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=stage1_iters)
                    simulation.context.setParameter("k", float(restraint_k2))
                    if stage2_iters > 0:
                        openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=stage2_iters)
            except AttributeError as _exc:
                if _is_nonetype_finalize_error(_exc):
                    raise RuntimeError(
                        f"OpenMM Context cleanup failed (restart {restart_idx}): "
                        f"NoneType _finalize bug. This restart skipped; other "
                        f"restarts may succeed."
                    ) from _exc
                raise

            # Get energy excluding TBMB restraint force group for clean dG_bind.
            # The TBMB restraints (group 10) are still active during minimization but
            # excluded from the reported energy.
            _physics_groups = ~(1 << _TBMB_FORCE_GROUP) if tbmb_restraints else -1
            st = simulation.context.getState(getEnergy=True, getPositions=True, groups=_physics_groups)
            minimized_positions = st.getPositions()
            minimized_positions_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            e_complex_kj_mol = float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

            # Log TBMB restraint energy for diagnostics (should be ~0 if within flat region).
            e_tbmb_restraint_kj = 0.0
            if tbmb_restraints:
                st_tbmb = simulation.context.getState(getEnergy=True, groups=(1 << _TBMB_FORCE_GROUP))
                e_tbmb_restraint_kj = float(st_tbmb.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
                if e_tbmb_restraint_kj > 1.0:
                    print(f"WARNING: TBMB restraint energy = {e_tbmb_restraint_kj:.2f} kJ/mol (restart {restart_idx})")

            # If energy is wildly out of range, retry with extra iterations (clash resolution).
            e_complex_kcal_mol = e_complex_kj_mol * KJ_MOL_TO_KCAL_MOL
            if (not math.isfinite(e_complex_kcal_mol)) or abs(e_complex_kcal_mol) > 5e4:
                extra_iters = max(200, int(stage2_iters) + 200)
                if min_protocol_norm == "twostage":
                    simulation.context.setParameter("k", float(restraint_k2))
                # R2-T03 (defect 3): same _finalize catcher as primary minimize calls
                try:
                    openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=int(extra_iters))
                except AttributeError as _exc:
                    if _is_nonetype_finalize_error(_exc):
                        raise RuntimeError(
                            f"OpenMM Context cleanup failed (restart {restart_idx}, extra-iter): "
                            f"NoneType _finalize bug. This restart skipped."
                        ) from _exc
                    raise
                st = simulation.context.getState(getEnergy=True, getPositions=True, groups=_physics_groups)
                minimized_positions = st.getPositions()
                minimized_positions_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                e_complex_kj_mol = float(st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
                e_complex_kcal_mol = e_complex_kj_mol * KJ_MOL_TO_KCAL_MOL
                if (not math.isfinite(e_complex_kcal_mol)) or abs(e_complex_kcal_mol) > 5e4:
                    raise ValueError(f"Unstable complex energy after minimization: {e_complex_kcal_mol:.3g} kcal/mol")

            minimized_raw = restart_dir / "complex_minimized.raw.pdb"
            with open(minimized_raw, "w") as f:
                app.PDBFile.writeFile(modeller.topology, minimized_positions, f, keepIds=True)

            # Ensure disulfide CONECT/SSBOND post-minimization.
            disulfides = detect_disulfide_bonds(str(minimized_raw))
            minimized_complex = restart_dir / "complex_minimized.pdb"
            add_cyclic_conect_records(
                str(minimized_raw),
                str(minimized_complex),
                disulfide_bonds=disulfides,
                covalent_bonds=ligand_covalent_bonds,
            )

            split = _split_complex_by_chain(str(minimized_complex), receptor_chain, ligand_chain, restart_dir)
            receptor_bonds = detect_disulfide_bonds(split.receptor_pdb)
            ligand_bonds = detect_disulfide_bonds(split.ligand_pdb)
            receptor_pdb = restart_dir / "receptor_minimized.pdb"
            ligand_pdb = restart_dir / "ligand_minimized.pdb"
            add_cyclic_conect_records(split.receptor_pdb, str(receptor_pdb), disulfide_bonds=receptor_bonds)
            # R2-T04 (defect 4): same head_to_tail filter for ligand PDB so
            # the standalone ligand-only createSystem below doesn't fail.
            _ligand_covalent_bonds_for_split = (
                filtered_ligand_covalent_bonds
                if (fix_structure and head_to_tail_present)
                else ligand_covalent_bonds
            )
            add_cyclic_conect_records(
                split.ligand_pdb,
                str(ligand_pdb),
                disulfide_bonds=ligand_bonds,
                covalent_bonds=_ligand_covalent_bonds_for_split,
            )

            def energy_kj_mol(pdb_file: Path) -> float:
                pdb_obj = app.PDBFile(str(pdb_file))
                sys_obj = forcefield.createSystem(
                    pdb_obj.topology,
                    nonbondedMethod=app.NoCutoff,
                    constraints=app.HBonds,
                )
                # R2-T04 (defect 4): if this is the ligand-only system and
                # head_to_tail was present, restore the amide HarmonicBondForce
                # post-createSystem (same trick as complex system above).
                if fix_structure and head_to_tail_present and str(pdb_file).endswith("ligand_minimized.pdb"):
                    _n1, _cN = _locate_head_to_tail_atom_indices(pdb_obj.topology, ligand_chain)
                    if _n1 is not None and _cN is not None:
                        _post_add_amide_bond_to_system(sys_obj, _n1, _cN)
                integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
                sim = app.Simulation(pdb_obj.topology, sys_obj, integ, platform_obj, properties)
                sim.context.setPositions(pdb_obj.positions)
                st2 = sim.context.getState(getEnergy=True)
                return float(st2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

            e_receptor_kj_mol = energy_kj_mol(receptor_pdb)
            e_ligand_kj_mol = energy_kj_mol(ligand_pdb)
            dg_bind_kj_mol = e_complex_kj_mol - e_receptor_kj_mol - e_ligand_kj_mol

            return {
                "restart_idx": int(restart_idx),
                "e_complex_kj_mol": e_complex_kj_mol,
                "e_receptor_kj_mol": e_receptor_kj_mol,
                "e_ligand_kj_mol": e_ligand_kj_mol,
                "dg_bind_kj_mol": dg_bind_kj_mol,
                "e_complex_kcal_mol": e_complex_kj_mol * KJ_MOL_TO_KCAL_MOL,
                "e_receptor_kcal_mol": e_receptor_kj_mol * KJ_MOL_TO_KCAL_MOL,
                "e_ligand_kcal_mol": e_ligand_kj_mol * KJ_MOL_TO_KCAL_MOL,
                "dg_bind_kcal_mol": dg_bind_kj_mol * KJ_MOL_TO_KCAL_MOL,
                "e_tbmb_restraint_kcal_mol": e_tbmb_restraint_kj * KJ_MOL_TO_KCAL_MOL,
                "_minimized_positions_nm": minimized_positions_nm,
                "minimized_complex_pdb": str(minimized_complex),
                "receptor_pdb": str(receptor_pdb),
                "ligand_pdb": str(ligand_pdb),
            }

        restart_records: List[Dict[str, Any]] = []
        restart_errors: List[Dict[str, Any]] = []
        last_restart_error: Optional[Exception] = None
        for r in range(n_restarts):
            try:
                restart_records.append(_minimize_and_score(r))
            except Exception as e:
                last_restart_error = e
                restart_errors.append({"restart_idx": int(r), "error": str(e)})
                continue

        if not restart_records:
            raise RuntimeError(f"All minimization restarts failed (n_restarts={n_restarts}): {last_restart_error}")

        restart_indices = [int(rec.get("restart_idx", i)) for i, rec in enumerate(restart_records)]
        dg_values = np.array([float(rec["dg_bind_kcal_mol"]) for rec in restart_records], dtype=float)
        if aggregate_norm == "best":
            agg_value = float(np.min(dg_values))
            chosen_idx = int(np.argmin(dg_values))
        elif aggregate_norm == "mean":
            agg_value = float(np.mean(dg_values))
            chosen_idx = int(np.argmin(np.abs(dg_values - agg_value)))
        else:
            agg_value = float(np.median(dg_values))
            chosen_idx = int(np.argmin(np.abs(dg_values - agg_value)))

        chosen = restart_records[chosen_idx]
        selected_restart_idx = int(chosen.get("restart_idx", chosen_idx))
        persist_paths = tmp_ctx is None

        def _compute_cross_terms_restarts() -> Tuple[Dict[str, float], Dict[str, List[float]]]:
            """
            Compute receptor↔ligand nonbonded interaction energies (LJ + Coulomb) at minimized coordinates.

            For stability under multi-start minimization, compute cross terms for every successful
            restart and aggregate them using the same `aggregate` rule as dg_bind.

            Returns:
              - aggregated cross terms (kcal/mol)
              - per-restart cross terms (lists, aligned with `restart_records`)
            """
            minimized_positions_nm_ref = np.asarray(chosen.get("_minimized_positions_nm"), dtype=float)
            rec_chain = _safe_chain_id(receptor_chain)
            lig_chain = _safe_chain_id(ligand_chain)

            # Atom indices for full-chain cross interaction.
            rec_atoms: List[int] = []
            lig_atoms: List[int] = []
            rec_atoms_heavy: List[int] = []
            lig_atoms_heavy: List[int] = []
            for atom in modeller.topology.atoms():
                chain_id = _safe_chain_id(getattr(atom.residue.chain, "id", ""))
                elem = atom.element.symbol.upper() if atom.element is not None else ""
                if chain_id == rec_chain:
                    rec_atoms.append(int(atom.index))
                    if elem != "H":
                        rec_atoms_heavy.append(int(atom.index))
                elif chain_id == lig_chain:
                    lig_atoms.append(int(atom.index))
                    if elem != "H":
                        lig_atoms_heavy.append(int(atom.index))

            if not rec_atoms or not lig_atoms:
                raise ValueError(
                    f"Failed to select receptor/ligand atoms for xint energy: "
                    f"receptor_chain={rec_chain!r} ({len(rec_atoms)} atoms), "
                    f"ligand_chain={lig_chain!r} ({len(lig_atoms)} atoms)"
                )

            # Heavy-atom-only cross interaction atom sets. This improves robustness of xint terms
            # to hydrogen placement / PDBFixer variability (critical for ranking metrics).
            if not rec_atoms_heavy:
                rec_atoms_heavy = list(rec_atoms)
            if not lig_atoms_heavy:
                lig_atoms_heavy = list(lig_atoms)

            # Interface-restricted atom sets:
            # identify interface residues by heavy-atom contacts within cutoff, then include all
            # atoms of those residues in the energy evaluation.
            iface_rec_atoms = set(rec_atoms)
            iface_lig_atoms = set(lig_atoms)
            iface_rec_atoms_heavy = set(rec_atoms_heavy)
            iface_lig_atoms_heavy = set(lig_atoms_heavy)
            iface_cutoff_nm = (
                float(cross_interaction_interface_cutoff_a) / 10.0 if cross_interaction_interface_cutoff_a > 0.0 else 0.0
            )
            if iface_cutoff_nm > 0.0 and cKDTree is not None:
                rec_res_atoms: Dict[int, List[int]] = {}
                lig_res_atoms: Dict[int, List[int]] = {}
                rec_res_heavy_atoms: Dict[int, List[int]] = {}
                lig_res_heavy_atoms: Dict[int, List[int]] = {}
                rec_heavy_coords: List[np.ndarray] = []
                lig_heavy_coords: List[np.ndarray] = []
                rec_heavy_res: List[int] = []
                lig_heavy_res: List[int] = []

                for chain in modeller.topology.chains():
                    chain_id = _safe_chain_id(getattr(chain, "id", ""))
                    if chain_id not in {rec_chain, lig_chain}:
                        continue
                    for res in chain.residues():
                        rid = int(res.index)
                        atom_list = rec_res_atoms if chain_id == rec_chain else lig_res_atoms
                        heavy_atom_list = rec_res_heavy_atoms if chain_id == rec_chain else lig_res_heavy_atoms
                        for a in res.atoms():
                            ai = int(a.index)
                            atom_list.setdefault(rid, []).append(ai)
                            elem = a.element.symbol.upper() if a.element is not None else ""
                            if elem and elem != "H":
                                heavy_atom_list.setdefault(rid, []).append(ai)
                                if chain_id == rec_chain:
                                    rec_heavy_coords.append(minimized_positions_nm_ref[ai])
                                    rec_heavy_res.append(rid)
                                else:
                                    lig_heavy_coords.append(minimized_positions_nm_ref[ai])
                                    lig_heavy_res.append(rid)

                if rec_heavy_coords and lig_heavy_coords:
                    rec_arr = np.asarray(rec_heavy_coords, dtype=float)
                    lig_arr = np.asarray(lig_heavy_coords, dtype=float)
                    tree_lig = cKDTree(lig_arr)
                    neigh_rec = tree_lig.query_ball_point(rec_arr, r=float(iface_cutoff_nm))
                    rec_iface_res = {int(rec_heavy_res[i]) for i, hits in enumerate(neigh_rec) if hits}

                    tree_rec = cKDTree(rec_arr)
                    neigh_lig = tree_rec.query_ball_point(lig_arr, r=float(iface_cutoff_nm))
                    lig_iface_res = {int(lig_heavy_res[i]) for i, hits in enumerate(neigh_lig) if hits}

                    if rec_iface_res and lig_iface_res:
                        iface_rec_atoms = {ai for rid in rec_iface_res for ai in rec_res_atoms.get(int(rid), [])}
                        iface_lig_atoms = {ai for rid in lig_iface_res for ai in lig_res_atoms.get(int(rid), [])}
                        iface_rec_atoms_heavy = {
                            ai for rid in rec_iface_res for ai in rec_res_heavy_atoms.get(int(rid), [])
                        }
                        iface_lig_atoms_heavy = {
                            ai for rid in lig_iface_res for ai in lig_res_heavy_atoms.get(int(rid), [])
                        }
                        if not iface_rec_atoms:
                            iface_rec_atoms = set(rec_atoms)
                        if not iface_lig_atoms:
                            iface_lig_atoms = set(lig_atoms)
                        if not iface_rec_atoms_heavy:
                            iface_rec_atoms_heavy = set(rec_atoms_heavy)
                        if not iface_lig_atoms_heavy:
                            iface_lig_atoms_heavy = set(lig_atoms_heavy)

            # Mutation-local cross interaction: restrict both sides to atoms within the
            # same neighborhood used for local backbone DOF / jitter localization.
            local_group = 4
            local_enabled = bool(ref_atom_indices) and float(restraint_exclusion_distance_a) > 0.0
            local_rec_atoms: set[int] = set()
            local_lig_atoms: set[int] = set()
            local_rec_atoms_heavy: set[int] = set()
            local_lig_atoms_heavy: set[int] = set()
            if local_enabled:
                thr2 = (float(restraint_exclusion_distance_a) / 10.0) ** 2
                ref_coords = minimized_positions_nm_ref[np.asarray(ref_atom_indices, dtype=int), :]
                if ref_coords.size > 0:
                    rec_idx = np.asarray(rec_atoms, dtype=int)
                    lig_idx = np.asarray(lig_atoms, dtype=int)
                    rec_coords = minimized_positions_nm_ref[rec_idx, :]
                    lig_coords = minimized_positions_nm_ref[lig_idx, :]

                    rec_diffs = rec_coords[:, None, :] - ref_coords[None, :, :]
                    lig_diffs = lig_coords[:, None, :] - ref_coords[None, :, :]
                    rec_min_d2 = np.min(np.sum(rec_diffs * rec_diffs, axis=-1), axis=1)
                    lig_min_d2 = np.min(np.sum(lig_diffs * lig_diffs, axis=-1), axis=1)

                    local_rec_atoms = {int(rec_idx[int(i)]) for i in np.where(rec_min_d2 < thr2)[0].tolist()}
                    local_lig_atoms = {int(lig_idx[int(i)]) for i in np.where(lig_min_d2 < thr2)[0].tolist()}
                if not local_rec_atoms:
                    local_rec_atoms = set(rec_atoms)
                if not local_lig_atoms:
                    local_lig_atoms = set(lig_atoms)

                # Heavy-only neighborhood (drop hydrogens from both ref and candidate atoms).
                heavy_set = set(int(x) for x in rec_atoms_heavy) | set(int(x) for x in lig_atoms_heavy)
                ref_heavy = [int(i) for i in ref_atom_indices if int(i) in heavy_set]
                if not ref_heavy:
                    ref_heavy = [int(i) for i in ref_atom_indices]
                ref_coords_h = minimized_positions_nm_ref[np.asarray(ref_heavy, dtype=int), :]
                if ref_coords_h.size > 0:
                    rec_idx_h = np.asarray(rec_atoms_heavy, dtype=int)
                    lig_idx_h = np.asarray(lig_atoms_heavy, dtype=int)
                    rec_coords_h = minimized_positions_nm_ref[rec_idx_h, :]
                    lig_coords_h = minimized_positions_nm_ref[lig_idx_h, :]

                    rec_diffs_h = rec_coords_h[:, None, :] - ref_coords_h[None, :, :]
                    lig_diffs_h = lig_coords_h[:, None, :] - ref_coords_h[None, :, :]
                    rec_min_d2_h = np.min(np.sum(rec_diffs_h * rec_diffs_h, axis=-1), axis=1)
                    lig_min_d2_h = np.min(np.sum(lig_diffs_h * lig_diffs_h, axis=-1), axis=1)

                    local_rec_atoms_heavy = {int(rec_idx_h[int(i)]) for i in np.where(rec_min_d2_h < thr2)[0].tolist()}
                    local_lig_atoms_heavy = {int(lig_idx_h[int(i)]) for i in np.where(lig_min_d2_h < thr2)[0].tolist()}
                if not local_rec_atoms_heavy:
                    local_rec_atoms_heavy = set(rec_atoms_heavy)
                if not local_lig_atoms_heavy:
                    local_lig_atoms_heavy = set(lig_atoms_heavy)

            # Extract per-particle nonbonded parameters from the complex system.
            nbf = None
            for f in system.getForces():
                if isinstance(f, openmm.NonbondedForce):
                    nbf = f
                    break
            if nbf is None:
                raise RuntimeError("NonbondedForce not found in OpenMM System; cannot compute xint energies")

            n_particles = int(system.getNumParticles())
            charges: List[float] = []
            sigmas: List[float] = []
            epsilons: List[float] = []
            for i in range(n_particles):
                q, sig, eps = nbf.getParticleParameters(i)
                charges.append(float(q.value_in_unit(unit.elementary_charge)))
                sigmas.append(float(sig.value_in_unit(unit.nanometer)))
                epsilons.append(float(eps.value_in_unit(unit.kilojoules_per_mole)))

            expr = (
                "4*sqrt(epsilon1*epsilon2)*("
                "((0.5*(sigma1+sigma2))/r)^12 - ((0.5*(sigma1+sigma2))/r)^6"
                ")"
                f" + ({_ONE_4PI_EPS0}*charge1*charge2)/(epsr*sqrt(r*r + rs*rs))"
            )

            soft_group = 3
            soft_enabled = iface_cutoff_nm > 0.0
            soft_width_nm = 0.10  # 1.0 A smoothing window around the interface cutoff
            soft_ron_nm = float(max(0.0, float(iface_cutoff_nm) - float(soft_width_nm)))
            soft_roff_nm = float(float(iface_cutoff_nm) + float(soft_width_nm))
            if soft_roff_nm <= soft_ron_nm:
                soft_enabled = False

            soft_expr = ""
            if soft_enabled:
                # Raised-cosine switching function:
                #   switch(r)=1 for r<=ron
                #   switch(r)=0 for r>=roff
                #   smooth in between
                pi = 3.141592653589793
                sw = (
                    f"step(ron - r) + step(r - ron)*step(roff - r)*"
                    f"(0.5*(1+cos({pi}*(r-ron)/(roff-ron))))"
                )
                soft_expr = f"({sw})*({expr})"

            def _make_force(
                *,
                expr_in: str,
                set1: set[int],
                set2: set[int],
                group: int,
                enable_soft: bool = False,
            ) -> openmm.CustomNonbondedForce:
                force = openmm.CustomNonbondedForce(expr_in)
                force.addPerParticleParameter("charge")
                force.addPerParticleParameter("sigma")
                force.addPerParticleParameter("epsilon")
                force.addGlobalParameter("rs", 0.0)  # nm
                force.addGlobalParameter("epsr", 1.0)
                if enable_soft:
                    force.addGlobalParameter("ron", float(soft_ron_nm))
                    force.addGlobalParameter("roff", float(soft_roff_nm))
                force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
                for i in range(n_particles):
                    force.addParticle([charges[i], sigmas[i], epsilons[i]])
                force.addInteractionGroup(set1, set2)
                force.setForceGroup(int(group))
                return force

            cross_sys = openmm.System()
            for i in range(n_particles):
                cross_sys.addParticle(system.getParticleMass(i))

            total_group = 1
            iface_group = 2
            total_heavy_group = 5
            iface_heavy_group = 6
            soft_heavy_group = 7
            local_heavy_group = 8
            cross_sys.addForce(_make_force(expr_in=expr, set1=set(rec_atoms), set2=set(lig_atoms), group=total_group))
            cross_sys.addForce(
                _make_force(expr_in=expr, set1=set(iface_rec_atoms), set2=set(iface_lig_atoms), group=iface_group)
            )
            cross_sys.addForce(
                _make_force(
                    expr_in=expr,
                    set1=set(rec_atoms_heavy),
                    set2=set(lig_atoms_heavy),
                    group=total_heavy_group,
                )
            )
            cross_sys.addForce(
                _make_force(
                    expr_in=expr,
                    set1=set(iface_rec_atoms_heavy),
                    set2=set(iface_lig_atoms_heavy),
                    group=iface_heavy_group,
                )
            )
            if soft_enabled:
                cross_sys.addForce(
                    _make_force(
                        expr_in=soft_expr,
                        set1=set(rec_atoms),
                        set2=set(lig_atoms),
                        group=soft_group,
                        enable_soft=True,
                    )
                )
                cross_sys.addForce(
                    _make_force(
                        expr_in=soft_expr,
                        set1=set(rec_atoms_heavy),
                        set2=set(lig_atoms_heavy),
                        group=soft_heavy_group,
                        enable_soft=True,
                    )
                )
            if local_enabled:
                cross_sys.addForce(
                    _make_force(
                        expr_in=expr,
                        set1=set(local_rec_atoms),
                        set2=set(local_lig_atoms),
                        group=local_group,
                    )
                )
                cross_sys.addForce(
                    _make_force(
                        expr_in=expr,
                        set1=set(local_rec_atoms_heavy),
                        set2=set(local_lig_atoms_heavy),
                        group=local_heavy_group,
                    )
                )

            integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
            ctx = openmm.Context(cross_sys, integ, platform_obj, properties)

            def _energy_kcal(group: int, *, rs_nm: float, epsr: float) -> float:
                ctx.setParameter("rs", float(rs_nm))
                ctx.setParameter("epsr", float(epsr))
                st_e = ctx.getState(getEnergy=True, groups=(1 << int(group)))
                e_kj = float(st_e.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
                return float(e_kj * KJ_MOL_TO_KCAL_MOL)

            def _terms_for_positions(pos_nm: np.ndarray) -> Dict[str, float]:
                ctx.setPositions(pos_nm * unit.nanometer)
                e_total = _energy_kcal(total_group, rs_nm=0.0, epsr=1.0)
                e_total_screened = _energy_kcal(
                    total_group,
                    rs_nm=float(coulomb_screening_distance_nm),
                    epsr=float(coulomb_screening_dielectric),
                )
                e_iface = _energy_kcal(iface_group, rs_nm=0.0, epsr=1.0)
                e_iface_screened = _energy_kcal(
                    iface_group,
                    rs_nm=float(coulomb_screening_distance_nm),
                    epsr=float(coulomb_screening_dielectric),
                )
                e_total_heavy = _energy_kcal(total_heavy_group, rs_nm=0.0, epsr=1.0)
                e_total_screened_heavy = _energy_kcal(
                    total_heavy_group,
                    rs_nm=float(coulomb_screening_distance_nm),
                    epsr=float(coulomb_screening_dielectric),
                )
                e_iface_heavy = _energy_kcal(iface_heavy_group, rs_nm=0.0, epsr=1.0)
                e_iface_screened_heavy = _energy_kcal(
                    iface_heavy_group,
                    rs_nm=float(coulomb_screening_distance_nm),
                    epsr=float(coulomb_screening_dielectric),
                )
                e_soft_iface = float("nan")
                e_soft_iface_screened = float("nan")
                e_soft_iface_heavy = float("nan")
                e_soft_iface_screened_heavy = float("nan")
                if soft_enabled:
                    e_soft_iface = _energy_kcal(soft_group, rs_nm=0.0, epsr=1.0)
                    e_soft_iface_screened = _energy_kcal(
                        soft_group,
                        rs_nm=float(coulomb_screening_distance_nm),
                        epsr=float(coulomb_screening_dielectric),
                    )
                    e_soft_iface_heavy = _energy_kcal(soft_heavy_group, rs_nm=0.0, epsr=1.0)
                    e_soft_iface_screened_heavy = _energy_kcal(
                        soft_heavy_group,
                        rs_nm=float(coulomb_screening_distance_nm),
                        epsr=float(coulomb_screening_dielectric),
                    )
                e_local = float("nan")
                e_local_screened = float("nan")
                e_local_heavy = float("nan")
                e_local_screened_heavy = float("nan")
                if local_enabled:
                    e_local = _energy_kcal(local_group, rs_nm=0.0, epsr=1.0)
                    e_local_screened = _energy_kcal(
                        local_group,
                        rs_nm=float(coulomb_screening_distance_nm),
                        epsr=float(coulomb_screening_dielectric),
                    )
                    e_local_heavy = _energy_kcal(local_heavy_group, rs_nm=0.0, epsr=1.0)
                    e_local_screened_heavy = _energy_kcal(
                        local_heavy_group,
                        rs_nm=float(coulomb_screening_distance_nm),
                        epsr=float(coulomb_screening_dielectric),
                    )
                return {
                    "e_cross_total_kcal_mol": float(e_total),
                    "e_cross_total_screened_kcal_mol": float(e_total_screened),
                    "e_cross_interface_total_kcal_mol": float(e_iface),
                    "e_cross_interface_total_screened_kcal_mol": float(e_iface_screened),
                    "e_cross_interface_soft_total_kcal_mol": float(e_soft_iface),
                    "e_cross_interface_soft_total_screened_kcal_mol": float(e_soft_iface_screened),
                    "e_cross_local_total_kcal_mol": float(e_local),
                    "e_cross_local_total_screened_kcal_mol": float(e_local_screened),
                    "e_cross_total_heavy_kcal_mol": float(e_total_heavy),
                    "e_cross_total_screened_heavy_kcal_mol": float(e_total_screened_heavy),
                    "e_cross_interface_total_heavy_kcal_mol": float(e_iface_heavy),
                    "e_cross_interface_total_screened_heavy_kcal_mol": float(e_iface_screened_heavy),
                    "e_cross_interface_soft_total_heavy_kcal_mol": float(e_soft_iface_heavy),
                    "e_cross_interface_soft_total_screened_heavy_kcal_mol": float(e_soft_iface_screened_heavy),
                    "e_cross_local_total_heavy_kcal_mol": float(e_local_heavy),
                    "e_cross_local_total_screened_heavy_kcal_mol": float(e_local_screened_heavy),
                }

            per_restart: Dict[str, List[float]] = {
                "e_cross_total_kcal_mol": [],
                "e_cross_total_screened_kcal_mol": [],
                "e_cross_interface_total_kcal_mol": [],
                "e_cross_interface_total_screened_kcal_mol": [],
                "e_cross_interface_soft_total_kcal_mol": [],
                "e_cross_interface_soft_total_screened_kcal_mol": [],
                "e_cross_local_total_kcal_mol": [],
                "e_cross_local_total_screened_kcal_mol": [],
                "e_cross_total_heavy_kcal_mol": [],
                "e_cross_total_screened_heavy_kcal_mol": [],
                "e_cross_interface_total_heavy_kcal_mol": [],
                "e_cross_interface_total_screened_heavy_kcal_mol": [],
                "e_cross_interface_soft_total_heavy_kcal_mol": [],
                "e_cross_interface_soft_total_screened_heavy_kcal_mol": [],
                "e_cross_local_total_heavy_kcal_mol": [],
                "e_cross_local_total_screened_heavy_kcal_mol": [],
            }
            for rec in restart_records:
                try:
                    pos_nm = np.asarray(rec.get("_minimized_positions_nm"), dtype=float)
                    terms = _terms_for_positions(pos_nm)
                except Exception:
                    # Keep alignment with restart_records; NaNs are ignored during aggregation.
                    terms = {k: float("nan") for k in per_restart}
                for k, v in terms.items():
                    per_restart[k].append(float(v))

            def _aggregate(values: List[float]) -> float:
                arr = np.asarray(values, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return float("nan")
                if aggregate_norm == "best":
                    v = float(arr[chosen_idx])
                    if math.isfinite(v):
                        return float(v)
                    return float(np.min(finite))
                if aggregate_norm == "mean":
                    return float(np.mean(finite))
                return float(np.median(finite))

            aggregated = {k: _aggregate(v) for k, v in per_restart.items()}
            restarts_out = {f"{k}_restarts": [float(x) for x in v] for k, v in per_restart.items()}
            return aggregated, restarts_out

        cross_terms, cross_terms_restarts = _compute_cross_terms_restarts()

        return {
            "e_complex_kj_mol": float(chosen["e_complex_kj_mol"]),
            "e_receptor_kj_mol": float(chosen["e_receptor_kj_mol"]),
            "e_ligand_kj_mol": float(chosen["e_ligand_kj_mol"]),
            "dg_bind_kj_mol": float(chosen["dg_bind_kj_mol"])
            if aggregate_norm == "best"
            else float(agg_value / KJ_MOL_TO_KCAL_MOL),
            "e_complex_kcal_mol": float(chosen["e_complex_kcal_mol"]),
            "e_receptor_kcal_mol": float(chosen["e_receptor_kcal_mol"]),
            "e_ligand_kcal_mol": float(chosen["e_ligand_kcal_mol"]),
            "dg_bind_kcal_mol": float(agg_value),
            "minimized_complex_pdb": str(chosen["minimized_complex_pdb"]) if persist_paths else "",
            "receptor_pdb": str(chosen["receptor_pdb"]) if persist_paths else "",
            "ligand_pdb": str(chosen["ligand_pdb"]) if persist_paths else "",
            "platform": requested,
            "device_index": int(device_index),
            "requested_platform": platform,
            "precision": precision_norm,
            "min_protocol": min_protocol_norm,
            "stage1_iters": int(stage1_iters),
            "stage2_iters": int(stage2_iters),
            "restraint_k1": float(restraint_k1),
            "restraint_k2": float(restraint_k2),
            "cross_interaction_interface_cutoff_a": float(cross_interaction_interface_cutoff_a),
            "coulomb_screening_distance_nm": float(coulomb_screening_distance_nm),
            "coulomb_screening_dielectric": float(coulomb_screening_dielectric),
            "restraint_exclusion_residues": [str(x) for x in (restraint_exclusion_residues or [])],
            "restraint_exclusion_distance_a": float(restraint_exclusion_distance_a),
            "n_restarts": int(n_restarts),
            "restart_indices": list(restart_indices),
            "jitter_nm": float(jitter_nm),
            "jitter_selected_atom_count": int(len(jitter_atom_indices)),
            "ligand_base_jitter_backbone_nm": float(ligand_base_jitter_backbone_nm),
            "ligand_base_jitter_sidechain_nm": float(ligand_base_jitter_sidechain_nm),
            "ligand_base_jitter_local_radius_a": float(ligand_base_jitter_local_radius_a),
            "ligand_base_jitter_selected_atom_count": int(len(ligand_base_jitter_indices)),
            "local_backbone_restart_jitter_nm": float(local_backbone_restart_jitter_nm),
            "local_backbone_selected_atom_count": int(len(local_backbone_indices)),
            "aggregate": aggregate_norm,
            "random_seed": int(random_seed),
            "dg_bind_kcal_mol_restarts": [float(x) for x in dg_values.tolist()],
            "selected_restart_index": int(selected_restart_idx),
            "restart_errors": restart_errors,
            **cross_terms,
            **cross_terms_restarts,
        }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def count_interface_hbonds(
    pdb_path: str,
    receptor_chain: str,
    ligand_chain: str,
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 120.0,
) -> int:
    """Count interface hydrogen bonds using heavy-atom geometric criteria.

    A hydrogen bond is counted when a potential donor heavy atom (N) on one chain
    is within `distance_cutoff` Angstroms of a potential acceptor heavy atom (O, N)
    on the other chain, and the D-H...A angle exceeds `angle_cutoff` degrees.

    For structures without explicit hydrogens, falls back to donor-acceptor distance
    only (< 3.5 A between N/O donor and O/N acceptor across the interface).

    Returns the total count of interface H-bonds (bidirectional: rec->lig + lig->rec).
    """
    rec_chain = _safe_chain_id(receptor_chain)
    lig_chain = _safe_chain_id(ligand_chain)

    # Parse atoms from PDB
    rec_donors: List[Tuple[np.ndarray, List[np.ndarray]]] = []  # (donor_coord, [H_coords])
    rec_acceptors: List[np.ndarray] = []
    lig_donors: List[Tuple[np.ndarray, List[np.ndarray]]] = []
    lig_acceptors: List[np.ndarray] = []

    DONOR_NAMES = {"N", "NE", "NH1", "NH2", "NZ", "ND1", "ND2", "NE1", "NE2", "OG", "OG1", "OH", "NE1"}
    ACCEPTOR_NAMES = {"O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "ND1", "NE2", "SD"}

    atoms_by_res: Dict[Tuple[str, int], List[dict]] = {}

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atom_name = line[12:16].strip()
                chain = _safe_chain_id(line[21].strip())
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            if chain not in (rec_chain, lig_chain):
                continue

            coord = np.array([x, y, z])
            entry = {"name": atom_name, "coord": coord, "chain": chain, "resnum": resnum}
            atoms_by_res.setdefault((chain, resnum), []).append(entry)

            if chain == rec_chain:
                if atom_name in DONOR_NAMES:
                    rec_donors.append((coord, []))
                if atom_name in ACCEPTOR_NAMES:
                    rec_acceptors.append(coord)
            elif chain == lig_chain:
                if atom_name in DONOR_NAMES:
                    lig_donors.append((coord, []))
                if atom_name in ACCEPTOR_NAMES:
                    lig_acceptors.append(coord)

    # Collect H atoms attached to donors (for angle check)
    for (chain, resnum), atoms in atoms_by_res.items():
        h_atoms = [a for a in atoms if a["name"].startswith("H")]
        donor_list = rec_donors if chain == rec_chain else lig_donors
        for d_coord, h_list in donor_list:
            for h in h_atoms:
                # H within 1.2 A of donor = attached
                if np.linalg.norm(h["coord"] - d_coord) < 1.2:
                    h_list.append(h["coord"])

    has_hydrogens = any(h_list for _, h_list in rec_donors + lig_donors)
    cutoff2 = distance_cutoff ** 2
    cos_cutoff = math.cos(math.radians(angle_cutoff))
    count = 0

    def _check_hbonds(donors, acceptors):
        nonlocal count
        for d_coord, h_list in donors:
            for a_coord in acceptors:
                d2 = float(np.sum((d_coord - a_coord) ** 2))
                if d2 > cutoff2 or d2 < 1.0:
                    continue
                if has_hydrogens and h_list:
                    # Check D-H...A angle for each attached H
                    for h_coord in h_list:
                        dh = d_coord - h_coord
                        ha = a_coord - h_coord
                        dh_norm = np.linalg.norm(dh)
                        ha_norm = np.linalg.norm(ha)
                        if dh_norm < 1e-6 or ha_norm < 1e-6:
                            continue
                        cos_angle = float(np.dot(dh, ha) / (dh_norm * ha_norm))
                        if cos_angle <= cos_cutoff:  # angle > cutoff (cosine decreases)
                            count += 1
                            break  # one H-bond per donor-acceptor pair
                else:
                    # No hydrogens: use distance-only criterion
                    count += 1

    _check_hbonds(rec_donors, lig_acceptors)
    _check_hbonds(lig_donors, rec_acceptors)

    return count
