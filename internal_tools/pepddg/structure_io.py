"""PDB/CIF structure parsing for PepDDG upstream feature pipeline.

Unified parser for both PDB and CIF formats (Boltz-2 outputs CIF,
many tools output PDB).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Standard 3-letter to 1-letter amino acid mapping
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Protonation/disulfide variants (OpenMM-relaxed structures)
    "HID": "H", "HIE": "H", "HIP": "H", "HSE": "H", "HSD": "H",
    "CYX": "C", "CSS": "C",
}


@dataclass(frozen=True)
class AtomRecord:
    chain: str
    resnum: int
    resname: str
    atom_name: str
    coord: np.ndarray  # (3,)
    element: str

    def __eq__(self, other):
        if not isinstance(other, AtomRecord):
            return NotImplemented
        return (
            self.chain == other.chain
            and self.resnum == other.resnum
            and self.resname == other.resname
            and self.atom_name == other.atom_name
            and self.element == other.element
            and np.allclose(self.coord, other.coord, atol=1e-3)
        )

    def __hash__(self):
        return hash((self.chain, self.resnum, self.resname, self.atom_name, self.element))


def _is_hydrogen(atom_name: str, element: str) -> bool:
    """Check if an atom is hydrogen."""
    if element.strip().upper() == "H":
        return True
    name = atom_name.strip()
    if name.startswith("H") or (len(name) > 1 and name[0].isdigit() and name[1] == "H"):
        return True
    return False


def parse_pdb(path: str | Path, chain_ids: list[str] | None = None) -> list[AtomRecord]:
    """Parse ATOM records from a PDB file.

    Args:
        path: Path to PDB file.
        chain_ids: Optional list of chain IDs to include. None = all chains.

    Returns:
        List of AtomRecord for heavy atoms.
    """
    atoms: list[AtomRecord] = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain_ids and chain not in chain_ids:
                continue
            try:
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            # Derive element from columns 77-78 or atom name
            element = ""
            if len(line) >= 78:
                element = line[76:78].strip()
            if not element:
                element = atom_name.lstrip("0123456789")[0] if atom_name else "X"
            if _is_hydrogen(atom_name, element):
                continue
            atoms.append(AtomRecord(
                chain=chain,
                resnum=resnum,
                resname=resname,
                atom_name=atom_name,
                coord=np.array([x, y, z], dtype=np.float64),
                element=element,
            ))
    return atoms


def parse_cif(path: str | Path, chain_ids: list[str] | None = None) -> list[AtomRecord]:
    """Parse atom records from an mmCIF file.

    Args:
        path: Path to CIF file.
        chain_ids: Optional list of chain IDs to include. None = all chains.

    Returns:
        List of AtomRecord for heavy atoms.
    """
    atoms: list[AtomRecord] = []
    path = Path(path)

    # Simple mmCIF loop parser — handles the _atom_site loop
    lines = path.read_text().splitlines()
    in_loop = False
    columns: list[str] = []
    data_rows: list[list[str]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "loop_":
            in_loop = True
            columns = []
            data_rows = []
            i += 1
            continue
        if in_loop and line.startswith("_atom_site."):
            columns.append(line)
            i += 1
            continue
        if in_loop and columns and not line.startswith("_") and not line.startswith("#") and line:
            tokens = line.split()
            if len(tokens) == len(columns):
                data_rows.append(tokens)
            i += 1
            continue
        if in_loop and columns and data_rows:
            break  # End of loop
        in_loop = False
        i += 1

    if not columns or not data_rows:
        return atoms

    # Build column index map
    col_idx = {c: i for i, c in enumerate(columns)}

    def get(row, key, default=""):
        idx = col_idx.get(key)
        return row[idx] if idx is not None else default

    for row in data_rows:
        group = get(row, "_atom_site.group_PDB")
        if group != "ATOM":
            continue
        chain = get(row, "_atom_site.auth_asym_id")
        if chain_ids and chain not in chain_ids:
            continue
        try:
            resnum = int(get(row, "_atom_site.auth_seq_id"))
            resname = get(row, "_atom_site.label_comp_id")
            atom_name = get(row, "_atom_site.label_atom_id")
            x = float(get(row, "_atom_site.Cartn_x"))
            y = float(get(row, "_atom_site.Cartn_y"))
            z = float(get(row, "_atom_site.Cartn_z"))
            element = get(row, "_atom_site.type_symbol", "X")
        except (ValueError, IndexError):
            continue
        if _is_hydrogen(atom_name, element):
            continue
        atoms.append(AtomRecord(
            chain=chain,
            resnum=resnum,
            resname=resname,
            atom_name=atom_name,
            coord=np.array([x, y, z], dtype=np.float64),
            element=element,
        ))
    return atoms


def parse_structure(
    path: str | Path,
    chain_ids: list[str] | None = None,
    atom_filter: str = "heavy",
) -> list[AtomRecord]:
    """Unified entry point: dispatch to PDB or CIF parser based on extension.

    Args:
        path: Path to structure file (.pdb or .cif).
        chain_ids: Optional chain filter.
        atom_filter: "heavy" (default) excludes hydrogens.

    Returns:
        List of AtomRecord.

    Raises:
        ValueError: If file extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        atoms = parse_pdb(path, chain_ids)
    elif suffix in (".cif", ".mmcif"):
        atoms = parse_cif(path, chain_ids)
    else:
        raise ValueError(f"Unsupported structure format: {suffix}")

    # Note: sub-parsers already exclude hydrogens, so atom_filter="heavy"
    # (the default) requires no additional filtering.
    return atoms


def get_residue_coord(
    atoms: list[AtomRecord],
    chain_id: str,
    resnum: int,
    prefer: str = "CB",
    fallback: str = "CA",
) -> np.ndarray | None:
    """Get coordinate of a specific atom in a residue, with fallback.

    Args:
        atoms: Parsed atom records.
        chain_id: Chain identifier.
        resnum: Residue number.
        prefer: Preferred atom name (e.g. "CB").
        fallback: Fallback atom name (e.g. "CA" for GLY).

    Returns:
        (3,) coordinate array, or None if residue not found.
    """
    prefer_coord = None
    fallback_coord = None
    for a in atoms:
        if a.chain == chain_id and a.resnum == resnum:
            if a.atom_name == prefer:
                prefer_coord = a.coord
            elif a.atom_name == fallback:
                fallback_coord = a.coord
    if prefer_coord is not None:
        return prefer_coord
    return fallback_coord


def get_chain_sequence(atoms: list[AtomRecord], chain_id: str) -> dict[int, str]:
    """Extract resnum -> 1-letter amino acid mapping from CA atoms.

    Args:
        atoms: Parsed atom records.
        chain_id: Chain identifier.

    Returns:
        Dict mapping residue number to 1-letter amino acid code.
    """
    residues: dict[int, str] = {}
    for a in atoms:
        if a.chain == chain_id and a.atom_name == "CA":
            aa = THREE_TO_ONE.get(a.resname, "X")
            residues[a.resnum] = aa
    return residues
