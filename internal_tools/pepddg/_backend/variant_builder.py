from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


AA1_TO_AA3: Dict[str, str] = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

AA3_TO_AA1: Dict[str, str] = {v: k for k, v in AA1_TO_AA3.items()}


def _safe_chain_id(chain: str) -> str:
    chain_id = (chain or "").strip()
    return chain_id if chain_id else "A"


_SKEMPI_MUT_RE = re.compile(r"^([A-Z])([A-Za-z])(\d+)([A-Z])$")


def parse_skempi_mutation_token(token: Optional[str]) -> Optional[Tuple[str, str, int, str]]:
    """
    Parse SKEMPI-style single-substitution tokens like: "TP6K" -> ("T", "P", 6, "K").

    Returns None for empty/NaN-like tokens or unsupported formats (e.g., indels/multi-muts).
    """
    if token is None:
        return None
    s = str(token).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    m = _SKEMPI_MUT_RE.match(s)
    if not m:
        return None
    wt, chain_id, resnum_s, mut = m.groups()
    chain_id = _safe_chain_id(chain_id.strip().upper())
    try:
        resnum = int(resnum_s)
    except Exception:
        return None
    if wt not in AA1_TO_AA3 or mut not in AA1_TO_AA3:
        return None
    return wt, chain_id, resnum, mut


def _get_residue_aa1(pdb_path: str, chain_id: str, resnum: int) -> Optional[str]:
    chain_id = _safe_chain_id(chain_id)
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue
            if _safe_chain_id(line[21].strip()) != chain_id:
                continue
            try:
                rnum = int(line[22:26].strip())
            except Exception:
                continue
            if rnum != int(resnum):
                continue
            resname = line[17:20].strip().upper()
            return AA3_TO_AA1.get(resname)
    return None


@dataclass(frozen=True)
class VariantSpec:
    wt_aa1: str
    chain_id: str
    resnum: int
    mut_aa1: str

    @property
    def label(self) -> str:
        return f"{self.wt_aa1}{_safe_chain_id(self.chain_id)}{int(self.resnum)}{self.mut_aa1}"


def _normalize_mutations(
    mutation: "Optional[VariantSpec | Sequence[VariantSpec]]",
) -> Sequence[VariantSpec]:
    """Normalize mutation input to a sequence of VariantSpec.

    Accepts None (WT), a single VariantSpec, or a sequence of VariantSpec.
    Returns an empty sequence for WT, or a tuple of VariantSpec otherwise.
    """
    if mutation is None:
        return ()
    if isinstance(mutation, VariantSpec):
        return (mutation,)
    return tuple(mutation)


def _mutations_label(mutations: Sequence[VariantSpec]) -> str:
    """Generate a combined label for one or more mutations."""
    if not mutations:
        return "WT"
    return "_".join(m.label for m in mutations)


def build_variant_pdb(
    base_complex_pdb: str,
    mutation: "Optional[VariantSpec | Sequence[VariantSpec]]",
    out_dir: str,
    *,
    ph: float = 7.0,
    keep_water: bool = False,
    strict_wt_check: bool = True,
    preserve_terminal_caps: bool = False,
    preparation_seed: Optional[int] = None,
) -> str:
    """
    Build a WT or mutant complex PDB using PDBFixer.

    Supports single-point and multi-point mutations. For multi-point mutations,
    pass a list/tuple of VariantSpec objects. Mutations on different chains are
    applied in per-chain batches.

    This helper preserves disulfide bonds by re-adding `CONECT/SSBOND` records
    after PDBFixer, since OpenMM's PDB writer does not preserve `CONECT`.
    """
    from .cyclic_peptide_utils import add_cyclic_conect_records, detect_disulfide_bonds

    try:
        from pdbfixer import PDBFixer  # type: ignore
        from openmm.app import PDBFile  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PDBFixer + OpenMM are required for build_variant_pdb()") from e

    base_path = Path(base_complex_pdb)
    if not base_path.exists():
        raise FileNotFoundError(str(base_path))

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    mutations = _normalize_mutations(mutation)
    variant_name = _mutations_label(mutations)

    # Encode disulfides pre-fixer so it doesn't add thiol H on bonded Cys.
    pre_bonds = detect_disulfide_bonds(str(base_path))
    with_conect = out_dir_path / f"{variant_name}.base_with_conect.pdb"
    add_cyclic_conect_records(str(base_path), str(with_conect), disulfide_bonds=pre_bonds)

    # Validate WT residues for all mutations
    if mutations and strict_wt_check:
        for mut in mutations:
            observed = _get_residue_aa1(str(with_conect), mut.chain_id, mut.resnum)
            if observed != mut.wt_aa1:
                raise ValueError(
                    f"WT residue mismatch for {mut.label}: "
                    f"observed={observed or 'UNKNOWN'} in {base_complex_pdb}"
                )

    fixer = PDBFixer(filename=str(with_conect))
    if preserve_terminal_caps:
        from ..terminal_caps import preserve_caps_in_fixer
        preserve_caps_in_fixer(fixer,keep_water=keep_water)
    else:
        fixer.removeHeterogens(keepWater=bool(keep_water))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # Apply mutations grouped by chain (PDBFixer requires per-chain calls)
    if mutations:
        chains_to_muts: Dict[str, list] = {}
        for mut in mutations:
            cid = _safe_chain_id(mut.chain_id)
            chains_to_muts.setdefault(cid, []).append(mut)

        for chain_id, chain_muts in sorted(chains_to_muts.items()):
            mut_tokens = [
                f"{AA1_TO_AA3[m.wt_aa1]}-{int(m.resnum)}-{AA1_TO_AA3[m.mut_aa1]}"
                for m in chain_muts
            ]
            fixer.applyMutations(mut_tokens, chain_id)

    fixer.findMissingResidues()
    # Do not attempt to rebuild missing residues/loops. For many PDBs this can fail due to
    # modified residues or incomplete templates, and is not required for fast scoring.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    if preparation_seed is None:
        fixer.addMissingAtoms()
    else:
        fixer.addMissingAtoms(seed=int(preparation_seed))
    if preserve_terminal_caps:
        from openmm.app import Modeller, ForceField
        from ..terminal_caps import cap_residue_templates
        modeller = Modeller(fixer.topology,fixer.positions)
        modeller.addHydrogens(ForceField("amber14-all.xml","implicit/obc2.xml"),
                             pH=float(ph),residueTemplates=cap_residue_templates(modeller.topology))
        fixer.topology,fixer.positions = modeller.topology,modeller.positions
    else:
        fixer.addMissingHydrogens(float(ph))

    raw_out = out_dir_path / f"{variant_name}.fixed.pdb"
    with open(raw_out, "w", encoding="utf-8") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    # Re-add disulfide CONECT/SSBOND after fixer.
    post_bonds = detect_disulfide_bonds(str(raw_out))
    final_out = out_dir_path / f"{variant_name}.pdb"
    add_cyclic_conect_records(str(raw_out), str(final_out), disulfide_bonds=post_bonds)
    return str(final_out)


def filter_mutations_for_chain(
    muts: Sequence[VariantSpec],
    chain_id: str,
) -> Tuple[VariantSpec, ...]:
    """Return mutations that are on the requested chain ID."""
    wanted = _safe_chain_id(chain_id)
    return tuple(m for m in muts if _safe_chain_id(m.chain_id) == wanted)
