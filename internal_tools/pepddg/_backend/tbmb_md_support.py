"""
TBMB Bicyclic Peptide MD Support

Handles TBMB (1,3,5-tris(bromomethyl)benzene) crosslinked bicyclic peptides
in OpenMM MD simulations.

TBMB crosslinks 3 cysteine residues via thioether bonds (CYS SG — CH2 — ring).
Boltz-2 outputs TBMB as a separate chain (LIG) with HETATM records and CONECT
records linking each methyl carbon to the corresponding CYS SG.

Strategy (Tier 1 — restraint-based):
    1. Parse TBMB HETATM + CONECT to find crosslinked CYS residues
    2. Measure SG-SG distances from the predicted structure
    3. Strip TBMB HETATM (OpenMM ff14SB has no TBMB parameters)
    4. Remove HG from crosslinked CYS (thioether-bonded, no free thiol)
    5. Add flat-bottom harmonic distance restraints between SG pairs
       to maintain bicyclic geometry

This approximation is valid for relative comparisons between peptides
on the same TBMB scaffold (same 3 CYS positions, same linker).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TBMBCrosslink:
    """A single TBMB thioether crosslink: CYS SG — TBMB methyl C."""

    cys_chain: str
    cys_resnum: int
    cys_sg_serial: int
    tbmb_atom_name: str  # e.g. "C20"
    tbmb_serial: int
    distance: float  # SG — C distance in Angstroms


@dataclass
class TBMBInfo:
    """Detected TBMB crosslinker information from a Boltz-2 PDB."""

    lig_chain: str  # chain ID of TBMB ligand (usually "C")
    lig_resname: str  # residue name (usually "LIG")
    lig_resnum: int
    n_heavy_atoms: int  # should be 9 for correct TBMB
    crosslinks: List[TBMBCrosslink] = field(default_factory=list)
    sg_sg_distances: Dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.crosslinks) == 3 and self.n_heavy_atoms == 9

    @property
    def cys_residues(self) -> List[Tuple[str, int]]:
        return [(cl.cys_chain, cl.cys_resnum) for cl in self.crosslinks]


def detect_tbmb_crosslinks(pdb_path: str) -> Optional[TBMBInfo]:
    """Detect TBMB crosslinker from Boltz-2 PDB output.

    Looks for:
    1. HETATM records with resname "LIG" and all-carbon atoms
    2. CONECT records linking LIG atoms to CYS SG atoms
    3. Validates: 9 heavy atoms, 3 crosslinks to CYS SG

    Args:
        pdb_path: Path to Boltz-2 output PDB

    Returns:
        TBMBInfo if TBMB detected, None otherwise
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # Parse all atoms
    atoms_by_serial: Dict[int, dict] = {}
    lig_atoms: List[dict] = []

    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            serial = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            resnum = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else ""
        except (ValueError, IndexError):
            continue

        record = {
            "serial": serial,
            "atom": atom_name,
            "resname": resname,
            "chain": chain,
            "resnum": resnum,
            "coord": np.array([x, y, z]),
            "element": element,
            "is_hetatm": line.startswith("HETATM"),
        }
        atoms_by_serial[serial] = record

        if line.startswith("HETATM") and resname == "LIG":
            lig_atoms.append(record)

    if not lig_atoms:
        return None

    # Verify all LIG atoms are carbon (TBMB = trimethylbenzene, all C)
    non_carbon = [a for a in lig_atoms if not a["atom"].startswith("C")]
    if non_carbon:
        return None

    lig_chain = lig_atoms[0]["chain"]
    lig_resnum = lig_atoms[0]["resnum"]

    # Parse CONECT records — normalize to unordered pairs to handle reciprocals
    conect_pairs = set()
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        try:
            base = int(line[6:11].strip())
        except ValueError:
            continue
        for start in (11, 16, 21, 26):
            chunk = line[start : start + 5].strip()
            if not chunk:
                continue
            try:
                other = int(chunk)
            except ValueError:
                continue
            conect_pairs.add((min(base, other), max(base, other)))

    # Find crosslinks: CONECT between LIG atom and CYS SG
    lig_serials = {a["serial"] for a in lig_atoms}
    crosslinks = []

    for s1, s2 in conect_pairs:
        a1 = atoms_by_serial.get(s1)
        a2 = atoms_by_serial.get(s2)
        if not a1 or not a2:
            continue

        # One must be LIG, other must be CYS SG
        if s1 in lig_serials and s2 not in lig_serials:
            lig_atom, prot_atom = a1, a2
        elif s2 in lig_serials and s1 not in lig_serials:
            lig_atom, prot_atom = a2, a1
        else:
            continue

        if prot_atom["resname"] != "CYS" or prot_atom["atom"] != "SG":
            continue

        dist = float(np.linalg.norm(lig_atom["coord"] - prot_atom["coord"]))
        crosslinks.append(
            TBMBCrosslink(
                cys_chain=prot_atom["chain"],
                cys_resnum=prot_atom["resnum"],
                cys_sg_serial=prot_atom["serial"],
                tbmb_atom_name=lig_atom["atom"],
                tbmb_serial=lig_atom["serial"],
                distance=dist,
            )
        )

    if len(crosslinks) != 3:
        return None

    # Sort crosslinks by CYS residue number
    crosslinks.sort(key=lambda cl: cl.cys_resnum)

    # Compute SG-SG distances (the restraint targets)
    sg_sg_distances = {}
    for i in range(len(crosslinks)):
        for j in range(i + 1, len(crosslinks)):
            cl_i, cl_j = crosslinks[i], crosslinks[j]
            sg_i = atoms_by_serial[cl_i.cys_sg_serial]
            sg_j = atoms_by_serial[cl_j.cys_sg_serial]
            dist = float(np.linalg.norm(sg_i["coord"] - sg_j["coord"]))
            key = f"SG_{cl_i.cys_chain}{cl_i.cys_resnum}-SG_{cl_j.cys_chain}{cl_j.cys_resnum}"
            sg_sg_distances[key] = dist

    return TBMBInfo(
        lig_chain=lig_chain,
        lig_resname="LIG",
        lig_resnum=lig_resnum,
        n_heavy_atoms=len(lig_atoms),
        crosslinks=crosslinks,
        sg_sg_distances=sg_sg_distances,
    )


def prepare_tbmb_for_md(
    pdb_path: str,
    output_dir: str,
    peptide_chain: str = "B",
    force_constant: float = 1000.0,
    flat_bottom_width: float = 1.5,
) -> Tuple[str, "TBMBInfo", list]:
    """Prepare a TBMB bicyclic peptide complex for MD simulation.

    Steps:
    1. Detect TBMB crosslinks from CONECT records
    2. Strip TBMB HETATM records (chain C LIG)
    3. Remove HG atoms from crosslinked CYS residues
    4. Generate flat-bottom distance restraints between SG pairs

    Args:
        pdb_path: Path to Boltz-2 output PDB
        output_dir: Directory for output files
        peptide_chain: Peptide chain ID
        force_constant: Restraint force constant (kJ/mol/nm^2)
        flat_bottom_width: Flat-bottom half-width in Angstroms

    Returns:
        Tuple of (cleaned_pdb_path, TBMBInfo, list_of_DistanceRestraint_dicts)

    Raises:
        ValueError: If TBMB crosslinks not detected or invalid
    """
    from .openmm_runner import DistanceRestraint

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect TBMB
    tbmb_info = detect_tbmb_crosslinks(pdb_path)
    if tbmb_info is None:
        raise ValueError(
            f"No valid TBMB crosslinker detected in {pdb_path}. "
            "Expected: 9 carbon HETATM atoms (LIG) with 3 CONECT to CYS SG."
        )

    if not tbmb_info.is_valid:
        raise ValueError(
            f"Invalid TBMB: {tbmb_info.n_heavy_atoms} heavy atoms "
            f"(expected 9), {len(tbmb_info.crosslinks)} crosslinks (expected 3)"
        )

    crosslinked_cys = {
        (cl.cys_chain, cl.cys_resnum) for cl in tbmb_info.crosslinks
    }

    # Step 2-3: Strip TBMB HETATM + remove HG from crosslinked CYS
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    lig_chain = tbmb_info.lig_chain

    # Pre-compute LIG serials + crosslinked CYS SG serials for CONECT filtering
    lig_serials = {a["serial"] for a in _get_lig_atoms(lines)}
    crosslink_sg_serials = {cl.cys_sg_serial for cl in tbmb_info.crosslinks}
    tbmb_related_serials = lig_serials | crosslink_sg_serials

    for line in lines:
        # Skip TBMB HETATM
        if line.startswith("HETATM"):
            chain = line[21].strip() or "A"
            resname = line[17:20].strip()
            if chain == lig_chain and resname == "LIG":
                continue

        # Skip CONECT records involving TBMB atoms
        if line.startswith("CONECT"):
            try:
                base = int(line[6:11].strip())
            except ValueError:
                cleaned_lines.append(line)
                continue
            # Check all referenced serials
            serials = [base]
            for start in (11, 16, 21, 26):
                chunk = line[start : start + 5].strip()
                if chunk:
                    try:
                        serials.append(int(chunk))
                    except ValueError:
                        pass
            # Skip if any serial is a TBMB LIG atom
            if any(s in lig_serials for s in serials):
                continue

        # Remove HG from crosslinked CYS (thioether-bonded, no free thiol)
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if atom_name == "HG":
                chain = line[21].strip() or "A"
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                if resname == "CYS" and (chain, resnum) in crosslinked_cys:
                    continue

        cleaned_lines.append(line)

    cleaned_pdb = out / "tbmb_stripped.pdb"
    with open(cleaned_pdb, "w") as f:
        f.writelines(cleaned_lines)

    # Step 4: Generate SG-SG distance restraints
    restraints = []
    crosslinks = tbmb_info.crosslinks

    for i in range(len(crosslinks)):
        for j in range(i + 1, len(crosslinks)):
            cl_i, cl_j = crosslinks[i], crosslinks[j]
            key = f"SG_{cl_i.cys_chain}{cl_i.cys_resnum}-SG_{cl_j.cys_chain}{cl_j.cys_resnum}"
            target_dist = tbmb_info.sg_sg_distances.get(key)

            if target_dist is None:
                continue

            restraints.append(
                DistanceRestraint(
                    chain1=cl_i.cys_chain,
                    resnum1=cl_i.cys_resnum,
                    atom1="SG",
                    chain2=cl_j.cys_chain,
                    resnum2=cl_j.cys_resnum,
                    atom2="SG",
                    distance=target_dist,
                    force_constant=force_constant,
                    kind="tbmb_thioether",
                )
            )

    # Write restraint info for reference
    info_path = out / "tbmb_restraints.txt"
    with open(info_path, "w") as f:
        f.write("TBMB Crosslinker Restraints\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Crosslinked CYS residues:\n")
        for cl in crosslinks:
            f.write(
                f"  {cl.cys_chain}:{cl.cys_resnum} SG — {cl.tbmb_atom_name} "
                f"(thioether dist: {cl.distance:.2f} A)\n"
            )
        f.write(f"\nSG-SG distance restraints:\n")
        for r in restraints:
            f.write(
                f"  {r.chain1}:{r.resnum1} SG — {r.chain2}:{r.resnum2} SG "
                f"= {r.distance:.2f} A (k={r.force_constant} kJ/mol/nm^2)\n"
            )

    return str(cleaned_pdb), tbmb_info, restraints


def _get_lig_atoms(lines: list) -> list:
    """Extract LIG HETATM atom info from PDB lines."""
    atoms = []
    for line in lines:
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname == "LIG":
                try:
                    serial = int(line[6:11].strip())
                    atoms.append({"serial": serial})
                except ValueError:
                    continue
    return atoms


def apply_flat_bottom_restraints(
    system,
    topology,
    positions,
    restraints: list,
    flat_bottom_width: float = 1.5,
):
    """Apply flat-bottom harmonic distance restraints for TBMB geometry.

    Uses a flat-bottom potential: zero force within [d0 - w, d0 + w],
    harmonic outside. This allows natural breathing of the bicyclic ring
    while preventing unphysical expansion.

    Energy = k * max(0, |r - r0| - w)^2

    where r0 = target distance, w = flat_bottom_width, k = force_constant.

    Args:
        system: OpenMM System
        topology: OpenMM Topology
        positions: Atom positions
        restraints: List of DistanceRestraint with kind="tbmb_thioether"
        flat_bottom_width: Half-width of flat region in Angstroms
    """
    from openmm import CustomBondForce, unit

    # Flat-bottom harmonic: zero inside flat region, harmonic outside
    # r0 is in nm, w is in nm, k is in kJ/mol/nm^2
    force = CustomBondForce(
        "k * max(0, abs(r - r0) - w)^2"
    )
    force.addPerBondParameter("r0")  # target distance (nm)
    force.addPerBondParameter("w")  # flat-bottom half-width (nm)
    force.addPerBondParameter("k")  # force constant (kJ/mol/nm^2)

    from .openmm_runner import _find_atom

    if hasattr(positions, "value_in_unit"):
        pos_nm = np.array(positions.value_in_unit(unit.nanometer))
    else:
        pos_nm = np.array([vec.value_in_unit(unit.nanometer) for vec in positions])

    w_nm = flat_bottom_width / 10.0  # Angstroms to nm

    n_added = 0
    for restraint in restraints:
        if restraint.kind != "tbmb_thioether":
            continue

        atom1 = _find_atom(
            topology, restraint.chain1, restraint.resnum1, restraint.atom1
        )
        atom2 = _find_atom(
            topology, restraint.chain2, restraint.resnum2, restraint.atom2
        )

        if atom1 is None or atom2 is None:
            print(
                f"WARNING: Could not find atoms for TBMB restraint "
                f"{restraint.chain1}{restraint.resnum1}:{restraint.atom1}-"
                f"{restraint.chain2}{restraint.resnum2}:{restraint.atom2}"
            )
            continue

        if restraint.distance is not None:
            r0_nm = restraint.distance / 10.0
        else:
            r0_nm = float(
                np.linalg.norm(pos_nm[atom1.index] - pos_nm[atom2.index])
            )

        k_val = (
            restraint.force_constant
            if restraint.force_constant is not None
            else 1000.0
        )

        force.addBond(atom1.index, atom2.index, [r0_nm, w_nm, k_val])
        n_added += 1

    if n_added > 0:
        system.addForce(force)
        print(f"Added {n_added} TBMB flat-bottom distance restraints")

    return n_added
