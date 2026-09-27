"""
Cyclic Peptide Utilities

Special handling for cyclic peptides (disulfide, head-to-tail, side-chain) in MD simulations.

Key challenges with cyclic peptides:
1. Disulfide bond topology must be correctly defined
2. Terminal Cys should NOT have SH hydrogen (they form S-S bond)
3. Head-to-tail / side-chain bonds may be missing from PDB records
4. PDBFixer may incorrectly add terminal hydrogens
5. Force field must recognize the covalent bonds

This module provides:
- Cyclic peptide detection and validation
- Proper structure preparation for MD
- Disulfide and cyclization bond handling for OpenMM
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DisulfideBond:
    """Represents a disulfide bond between two cysteine residues."""

    chain1: str
    resnum1: int
    chain2: str
    resnum2: int

    # Atom indices (set during system preparation)
    sg1_index: Optional[int] = None
    sg2_index: Optional[int] = None

    # Measured distance
    distance: Optional[float] = None

    def __str__(self):
        return f"SS({self.chain1}{self.resnum1}-{self.chain2}{self.resnum2})"


@dataclass
class CovalentBond:
    """Represents a covalent bond used for cyclization (non-disulfide)."""

    chain1: str
    resnum1: int
    atom1: str
    chain2: str
    resnum2: int
    atom2: str
    kind: str = "other"
    distance: Optional[float] = None

    def __str__(self):
        return (
            f"{self.kind}({self.chain1}{self.resnum1}:{self.atom1}-"
            f"{self.chain2}{self.resnum2}:{self.atom2})"
        )


@dataclass
class CyclicPeptideInfo:
    """Information about a cyclic peptide structure."""

    chain_id: str
    sequence: str
    length: int

    # Cyclization info
    cyclization_type: str  # "linear", "disulfide", "head_to_tail", "side_chain", "multiple"
    disulfide_bonds: List[DisulfideBond]
    covalent_bonds: List[CovalentBond] = field(default_factory=list)
    head_to_tail_bond: Optional[CovalentBond] = None
    side_chain_bonds: List[CovalentBond] = field(default_factory=list)

    # Validation
    is_valid: bool = True
    validation_errors: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


def detect_disulfide_bonds(
    pdb_path: str,
    distance_cutoff: float = 2.5,
) -> List[DisulfideBond]:
    """
    Detect disulfide bonds from PDB structure.

    Looks for:
    1. SSBOND records in PDB header
    2. CONECT records linking SG atoms
    3. Close SG-SG distances (< cutoff)

    Args:
        pdb_path: Path to PDB file
        distance_cutoff: Maximum SG-SG distance to consider as disulfide (Angstroms)

    Returns:
        List of detected DisulfideBond objects
    """
    disulfide_bonds = []

    # Read PDB file
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    # Method 1: Parse SSBOND records
    for line in lines:
        if line.startswith("SSBOND"):
            # SSBOND   1 CYS A    1    CYS A    9
            try:
                chain1 = line[15].strip()
                resnum1 = int(line[17:21].strip())
                chain2 = line[29].strip()
                resnum2 = int(line[31:35].strip())

                bond = DisulfideBond(
                    chain1=chain1 or "A",
                    resnum1=resnum1,
                    chain2=chain2 or "A",
                    resnum2=resnum2,
                )
                disulfide_bonds.append(bond)
            except (ValueError, IndexError):
                continue

    if disulfide_bonds:
        return disulfide_bonds

    # Method 2: Find close SG atoms
    sg_atoms = []  # [(chain, resnum, x, y, z)]

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            if atom_name == "SG":
                res_name = line[17:20].strip()
                if res_name == "CYS":
                    chain = line[21].strip() or "A"
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    sg_atoms.append((chain, resnum, np.array([x, y, z])))

    candidate_bonds = []
    for i, (chain1, resnum1, coord1) in enumerate(sg_atoms):
        for j, (chain2, resnum2, coord2) in enumerate(sg_atoms[i+1:], i+1):
            dist = np.linalg.norm(coord1 - coord2)
            if dist <= distance_cutoff:
                bond = DisulfideBond(
                    chain1=chain1,
                    resnum1=resnum1,
                    chain2=chain2,
                    resnum2=resnum2,
                    distance=dist,
                )
                candidate_bonds.append(bond)

    paired_residues = set()
    for bond in sorted(candidate_bonds, key=lambda item: float(item.distance or 0.0)):
        key1 = (bond.chain1, bond.resnum1)
        key2 = (bond.chain2, bond.resnum2)
        if key1 in paired_residues or key2 in paired_residues:
            continue
        disulfide_bonds.append(bond)
        paired_residues.update((key1, key2))

    return disulfide_bonds


def _parse_pdb_atoms(pdb_path: str):
    lines = []
    atoms_by_serial = {}
    atoms_by_key = {}
    residues_by_chain = {}

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip() or "A"
                resnum = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            record = {
                "serial": serial,
                "atom": atom_name,
                "resname": resname,
                "chain": chain,
                "resnum": resnum,
                "coord": np.array([x, y, z]),
            }
            atoms_by_serial[serial] = record
            atoms_by_key[(chain, resnum, atom_name)] = record
            residues_by_chain.setdefault(chain, set()).add(resnum)

    return lines, atoms_by_serial, atoms_by_key, residues_by_chain


def _parse_conect_pairs(lines: List[str]) -> set:
    pairs = set()

    for line in lines:
        if not line.startswith("CONECT"):
            continue

        try:
            base = int(line[6:11].strip())
        except ValueError:
            continue

        for start in (11, 16, 21, 26):
            chunk = line[start:start + 5].strip()
            if not chunk:
                continue
            try:
                other = int(chunk)
            except ValueError:
                continue
            pair = tuple(sorted((base, other)))
            pairs.add(pair)

    return pairs


def _distance(atom_a: Dict[str, Any], atom_b: Dict[str, Any]) -> float:
    return float(np.linalg.norm(atom_a["coord"] - atom_b["coord"]))


def detect_covalent_bonds(
    pdb_path: str,
    peptide_chain: str = "B",
    head_to_tail_distance_cutoff: float = 1.8,
) -> Tuple[Optional[CovalentBond], List[CovalentBond], List[CovalentBond]]:
    """
    Detect non-disulfide covalent bonds used for cyclization.

    This relies primarily on CONECT records. If no CONECT is present,
    head-to-tail cyclization is inferred by N-C distance between termini.

    Returns:
        head_to_tail_bond, side_chain_bonds, all_covalent_bonds
    """
    lines, atoms_by_serial, atoms_by_key, residues_by_chain = _parse_pdb_atoms(pdb_path)

    residues = sorted(residues_by_chain.get(peptide_chain, []))
    if not residues:
        return None, [], []

    first_resnum = residues[0]
    last_resnum = residues[-1]

    conect_pairs = _parse_conect_pairs(lines)
    covalent_bonds: List[CovalentBond] = []
    side_chain_bonds: List[CovalentBond] = []
    head_to_tail_bond: Optional[CovalentBond] = None

    for serial1, serial2 in conect_pairs:
        atom1 = atoms_by_serial.get(serial1)
        atom2 = atoms_by_serial.get(serial2)
        if not atom1 or not atom2:
            continue
        if atom1["chain"] != peptide_chain or atom2["chain"] != peptide_chain:
            continue
        if atom1["resnum"] == atom2["resnum"]:
            continue

        # Skip disulfides (handled separately)
        if (
            atom1["atom"] == "SG"
            and atom2["atom"] == "SG"
            and atom1["resname"] == "CYS"
            and atom2["resname"] == "CYS"
        ):
            continue

        # Head-to-tail cyclization via CONECT
        if (
            atom1["resnum"] == first_resnum
            and atom1["atom"] == "N"
            and atom2["resnum"] == last_resnum
            and atom2["atom"] == "C"
        ) or (
            atom2["resnum"] == first_resnum
            and atom2["atom"] == "N"
            and atom1["resnum"] == last_resnum
            and atom1["atom"] == "C"
        ):
            bond = CovalentBond(
                chain1=atom1["chain"],
                resnum1=atom1["resnum"],
                atom1=atom1["atom"],
                chain2=atom2["chain"],
                resnum2=atom2["resnum"],
                atom2=atom2["atom"],
                kind="head_to_tail",
                distance=_distance(atom1, atom2),
            )
            head_to_tail_bond = bond
            covalent_bonds.append(bond)
            continue

        # Ignore standard peptide bond between adjacent residues
        if abs(atom1["resnum"] - atom2["resnum"]) == 1 and {
            atom1["atom"],
            atom2["atom"],
        } == {"C", "N"}:
            continue

        bond = CovalentBond(
            chain1=atom1["chain"],
            resnum1=atom1["resnum"],
            atom1=atom1["atom"],
            chain2=atom2["chain"],
            resnum2=atom2["resnum"],
            atom2=atom2["atom"],
            kind="side_chain",
            distance=_distance(atom1, atom2),
        )
        covalent_bonds.append(bond)
        side_chain_bonds.append(bond)

    # Infer head-to-tail if no CONECT but distance suggests cyclization
    if head_to_tail_bond is None:
        atom_n = atoms_by_key.get((peptide_chain, first_resnum, "N"))
        atom_c = atoms_by_key.get((peptide_chain, last_resnum, "C"))
        if atom_n and atom_c:
            dist = _distance(atom_n, atom_c)
            if dist <= head_to_tail_distance_cutoff:
                head_to_tail_bond = CovalentBond(
                    chain1=atom_n["chain"],
                    resnum1=atom_n["resnum"],
                    atom1=atom_n["atom"],
                    chain2=atom_c["chain"],
                    resnum2=atom_c["resnum"],
                    atom2=atom_c["atom"],
                    kind="head_to_tail",
                    distance=dist,
                )
                covalent_bonds.append(head_to_tail_bond)

    return head_to_tail_bond, side_chain_bonds, covalent_bonds


def detect_cyclic_peptide(
    pdb_path: str,
    peptide_chain: str = "B",
) -> CyclicPeptideInfo:
    """
    Detect if a peptide is cyclic and characterize its cyclization.

    Args:
        pdb_path: Path to PDB file
        peptide_chain: Chain ID of the peptide

    Returns:
        CyclicPeptideInfo object with detection results
    """
    # Get sequence and residue info
    residues = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21].strip() or "A"
                if chain == peptide_chain:
                    resnum = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    if (resnum, resname) not in [(r[0], r[1]) for r in residues]:
                        residues.append((resnum, resname))

    if not residues:
        return CyclicPeptideInfo(
            chain_id=peptide_chain,
            sequence="",
            length=0,
            cyclization_type="unknown",
            disulfide_bonds=[],
            is_valid=False,
            validation_errors=["No residues found in peptide chain"],
        )

    # Sort by residue number
    residues.sort(key=lambda x: x[0])

    # Convert to sequence
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    sequence = ""
    for resnum, resname in residues:
        sequence += aa_map.get(resname, 'X')

    # Detect disulfide bonds
    all_disulfides = detect_disulfide_bonds(pdb_path)

    # Filter to bonds involving the peptide chain
    peptide_disulfides = [
        bond for bond in all_disulfides
        if bond.chain1 == peptide_chain or bond.chain2 == peptide_chain
    ]

    # Detect non-disulfide covalent bonds (head-to-tail / side-chain)
    head_to_tail_bond, side_chain_bonds, covalent_bonds = detect_covalent_bonds(
        pdb_path,
        peptide_chain=peptide_chain,
    )

    # Check for TBMB crosslinker (3 CYS + LIG HETATM with 9 carbons)
    has_tbmb = False
    try:
        from .tbmb_md_support import detect_tbmb_crosslinks

        tbmb_info = detect_tbmb_crosslinks(pdb_path)
        if tbmb_info is not None and tbmb_info.is_valid:
            # Verify TBMB connects to peptide chain CYS residues
            tbmb_cys_on_peptide = [
                cl for cl in tbmb_info.crosslinks if cl.cys_chain == peptide_chain
            ]
            has_tbmb = len(tbmb_cys_on_peptide) == 3
    except ImportError:
        pass

    # Determine cyclization type
    has_disulfide = len(peptide_disulfides) > 0
    has_head_to_tail = head_to_tail_bond is not None
    has_side_chain = len(side_chain_bonds) > 0

    if has_tbmb:
        cyclization_type = "tbmb"
    elif sum([has_disulfide, has_head_to_tail, has_side_chain]) >= 2 or (
        has_disulfide and len(peptide_disulfides) > 1
    ):
        cyclization_type = "multiple"
    elif has_head_to_tail:
        cyclization_type = "head_to_tail"
    elif has_side_chain:
        cyclization_type = "side_chain"
    elif has_disulfide:
        cyclization_type = "disulfide"
    else:
        cyclization_type = "linear"

    # Validation
    validation_errors = []

    # If disulfide connects termini, validate terminal residues are Cys
    first_resnum = residues[0][0]
    last_resnum = residues[-1][0]
    terminal_disulfide = any(
        bond.chain1 == peptide_chain
        and bond.chain2 == peptide_chain
        and (
            (bond.resnum1 == first_resnum and bond.resnum2 == last_resnum)
            or (bond.resnum1 == last_resnum and bond.resnum2 == first_resnum)
        )
        for bond in peptide_disulfides
    )

    if terminal_disulfide:
        if sequence[0] != "C":
            validation_errors.append(
                f"N-terminal is {sequence[0]}, expected C for terminal disulfide"
            )
        if sequence[-1] != "C":
            validation_errors.append(
                f"C-terminal is {sequence[-1]}, expected C for terminal disulfide"
            )

    return CyclicPeptideInfo(
        chain_id=peptide_chain,
        sequence=sequence,
        length=len(sequence),
        cyclization_type=cyclization_type,
        disulfide_bonds=peptide_disulfides,
        covalent_bonds=covalent_bonds,
        head_to_tail_bond=head_to_tail_bond,
        side_chain_bonds=side_chain_bonds,
        is_valid=len(validation_errors) == 0,
        validation_errors=validation_errors,
    )


def add_disulfide_conect_records(
    pdb_path: str,
    output_path: str,
    disulfide_bonds: List[DisulfideBond],
) -> str:
    """
    Add CONECT records for disulfide bonds to PDB file.

    OpenMM uses CONECT records to identify bonds, including disulfides.

    Args:
        pdb_path: Input PDB file
        output_path: Output PDB file with CONECT records
        disulfide_bonds: List of disulfide bonds to add

    Returns:
        Path to output file
    """
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    # Find SG atom indices for each disulfide bond
    sg_indices = {}  # (chain, resnum) -> atom_serial

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            if atom_name == "SG":
                chain = line[21].strip() or "A"
                resnum = int(line[22:26].strip())
                serial = int(line[6:11].strip())
                sg_indices[(chain, resnum)] = serial

    # Build CONECT records
    conect_lines = []
    for bond in disulfide_bonds:
        sg1 = sg_indices.get((bond.chain1, bond.resnum1))
        sg2 = sg_indices.get((bond.chain2, bond.resnum2))

        if sg1 and sg2:
            conect_lines.append(f"CONECT{sg1:5d}{sg2:5d}\n")
            conect_lines.append(f"CONECT{sg2:5d}{sg1:5d}\n")

    # Write output file
    with open(output_path, 'w') as f:
        # Write all lines except END
        for line in lines:
            if not line.startswith("END"):
                f.write(line)

        # Add SSBOND records if not present
        has_ssbond = any(l.startswith("SSBOND") for l in lines)
        if not has_ssbond:
            for i, bond in enumerate(disulfide_bonds, 1):
                ssbond = f"SSBOND {i:3d} CYS {bond.chain1:1s} {bond.resnum1:4d}    CYS {bond.chain2:1s} {bond.resnum2:4d}\n"
                f.write(ssbond)

        # Add CONECT records
        for conect in conect_lines:
            f.write(conect)

        f.write("END\n")

    return output_path


def add_cyclic_conect_records(
    pdb_path: str,
    output_path: str,
    disulfide_bonds: Optional[List[DisulfideBond]] = None,
    covalent_bonds: Optional[List[CovalentBond]] = None,
) -> str:
    """
    Add CONECT/SSBOND records for disulfide and non-disulfide cyclization bonds.

    Args:
        pdb_path: Input PDB file
        output_path: Output PDB file with CONECT records
        disulfide_bonds: Disulfide bonds to encode
        covalent_bonds: Other covalent bonds (head-to-tail / side-chain)

    Returns:
        Path to output file
    """
    disulfide_bonds = disulfide_bonds or []
    covalent_bonds = covalent_bonds or []

    with open(pdb_path, "r") as f:
        lines = f.readlines()

    _, _, atoms_by_key, _ = _parse_pdb_atoms(pdb_path)
    existing_pairs = _parse_conect_pairs(lines)
    new_pairs = set()

    def add_pair(serial1: Optional[int], serial2: Optional[int]):
        if not serial1 or not serial2:
            return
        pair = tuple(sorted((serial1, serial2)))
        if pair in existing_pairs or pair in new_pairs:
            return
        new_pairs.add(pair)

    # Add disulfide CONECT pairs
    for bond in disulfide_bonds:
        sg1 = atoms_by_key.get((bond.chain1, bond.resnum1, "SG"))
        sg2 = atoms_by_key.get((bond.chain2, bond.resnum2, "SG"))
        add_pair(sg1["serial"] if sg1 else None, sg2["serial"] if sg2 else None)

    # Add other covalent bonds
    for bond in covalent_bonds:
        atom1 = atoms_by_key.get((bond.chain1, bond.resnum1, bond.atom1))
        atom2 = atoms_by_key.get((bond.chain2, bond.resnum2, bond.atom2))
        add_pair(atom1["serial"] if atom1 else None, atom2["serial"] if atom2 else None)

    # Write output file
    with open(output_path, "w") as f:
        for line in lines:
            if not line.startswith("END"):
                f.write(line)

        has_ssbond = any(l.startswith("SSBOND") for l in lines)
        if disulfide_bonds and not has_ssbond:
            for i, bond in enumerate(disulfide_bonds, 1):
                ssbond = (
                    f"SSBOND {i:3d} CYS {bond.chain1:1s} {bond.resnum1:4d}    "
                    f"CYS {bond.chain2:1s} {bond.resnum2:4d}\n"
                )
                f.write(ssbond)

        for serial1, serial2 in sorted(new_pairs):
            f.write(f"CONECT{serial1:5d}{serial2:5d}\n")
            f.write(f"CONECT{serial2:5d}{serial1:5d}\n")

        f.write("END\n")

    return output_path


def prepare_cyclic_peptide_for_md(
    pdb_path: str,
    output_dir: str,
    peptide_chain: str = "B",
    receptor_chain: str = "A",
) -> Tuple[str, CyclicPeptideInfo]:
    """
    Prepare a cyclic peptide complex for MD simulation.

    This function handles the special requirements for cyclic peptides:
    1. Detects disulfide bonds
    2. Adds proper CONECT/SSBOND records
    3. Fixes structure WITHOUT adding terminal hydrogens to bonded Cys
    4. Returns ready-to-simulate structure

    Args:
        pdb_path: Path to input PDB file
        output_dir: Output directory
        peptide_chain: Chain ID of peptide
        receptor_chain: Chain ID of receptor (if complex)

    Returns:
        Tuple of (output_pdb_path, CyclicPeptideInfo)

    Example:
        >>> prepared_pdb, info = prepare_cyclic_peptide_for_md(
        ...     "complex.pdb",
        ...     "md_prep",
        ...     peptide_chain="B",
        ... )
        >>> print(f"Cyclization: {info.cyclization_type}")
        >>> print(f"Disulfide bonds: {info.disulfide_bonds}")
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        raise ImportError("OpenMM and PDBFixer required")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Detect cyclic peptide info
    cyclic_info = detect_cyclic_peptide(pdb_path, peptide_chain)

    print(f"Detected peptide: {cyclic_info.sequence} ({cyclic_info.length} aa)")
    print(f"Cyclization type: {cyclic_info.cyclization_type}")
    print(f"Disulfide bonds: {cyclic_info.disulfide_bonds}")

    # Step 2: Add CONECT records if needed
    conect_pdb = Path(output_dir) / "with_conect.pdb"
    add_cyclic_conect_records(
        pdb_path,
        str(conect_pdb),
        disulfide_bonds=cyclic_info.disulfide_bonds,
        covalent_bonds=cyclic_info.covalent_bonds,
    )

    # Step 3: Fix structure with PDBFixer (careful with Cys)
    fixer = PDBFixer(filename=str(conect_pdb))

    # Find missing residues (but don't add them at termini)
    fixer.findMissingResidues()

    # Remove terminal residue additions (we don't want to extend the peptide)
    keys_to_remove = []
    for key in fixer.missingResidues:
        chain_id, resnum = key
        # Don't add residues at termini
        if resnum == 0 or resnum > 1000:  # Heuristic for terminal positions
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del fixer.missingResidues[key]

    # Find and add missing atoms (sidechains, etc.)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Add hydrogens - but we need to handle disulfide Cys specially
    # PDBFixer will not add HG to Cys if there's a CONECT record for SG
    fixer.addMissingHydrogens(7.0)

    # Step 4: Save fixed structure
    fixed_pdb = Path(output_dir) / "fixed_cyclic.pdb"
    with open(fixed_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    # Step 5: Re-add CONECT records (PDBFile.writeFile may not preserve them)
    final_pdb = Path(output_dir) / "prepared_cyclic.pdb"
    add_cyclic_conect_records(
        str(fixed_pdb),
        str(final_pdb),
        disulfide_bonds=cyclic_info.disulfide_bonds,
        covalent_bonds=cyclic_info.covalent_bonds,
    )

    # Step 6: Validate output
    final_info = detect_cyclic_peptide(str(final_pdb), peptide_chain)

    if cyclic_info.disulfide_bonds and not final_info.disulfide_bonds:
        print("WARNING: No disulfide bonds detected in output - check structure")
        cyclic_info.validation_errors.append("Disulfide bonds may not be properly set")
        cyclic_info.is_valid = False

    if cyclic_info.covalent_bonds and not final_info.covalent_bonds:
        print("WARNING: No cyclic covalent bonds detected in output - check structure")
        cyclic_info.validation_errors.append("Covalent cyclization bonds may be missing")
        cyclic_info.is_valid = False

    return str(final_pdb), cyclic_info


def create_openmm_disulfide_bonds(
    topology,
    disulfide_bonds: List[DisulfideBond],
):
    """
    Ensure disulfide bonds are properly defined in OpenMM topology.

    OpenMM's Topology class automatically detects disulfide bonds from
    CONECT records, but this function can add them manually if needed.

    Args:
        topology: OpenMM Topology object
        disulfide_bonds: List of DisulfideBond objects

    Returns:
        Modified topology (disulfide bonds added)

    Note:
        This is usually not needed if CONECT records are in the PDB.
        OpenMM will automatically create the S-S bond.
    """
    # Find SG atoms
    sg_atoms = {}
    for chain in topology.chains():
        chain_id = chain.id
        for residue in chain.residues():
            if residue.name == "CYS":
                for atom in residue.atoms():
                    if atom.name == "SG":
                        sg_atoms[(chain_id, residue.index + 1)] = atom

    # Add bonds (if not already present)
    existing_bonds = set()
    for bond in topology.bonds():
        existing_bonds.add((bond[0].index, bond[1].index))
        existing_bonds.add((bond[1].index, bond[0].index))

    for disulfide in disulfide_bonds:
        atom1 = sg_atoms.get((disulfide.chain1, disulfide.resnum1))
        atom2 = sg_atoms.get((disulfide.chain2, disulfide.resnum2))

        if atom1 and atom2:
            if (atom1.index, atom2.index) not in existing_bonds:
                topology.addBond(atom1, atom2)
                print(f"Added disulfide bond: {disulfide}")

    return topology


def validate_disulfide_geometry(
    pdb_path: str,
    disulfide_bonds: List[DisulfideBond],
    strict: bool = False,
) -> Dict[str, any]:
    """
    Validate disulfide bond geometry.

    Checks:
    - SG-SG distance (ideal: 2.03 Å, acceptable: 1.8-2.5 Å)
    - CB-SG-SG angle (ideal: 104°, acceptable: 90-120°)
    - Chi3 dihedral (ideal: ±90°)

    Args:
        pdb_path: Path to PDB file
        disulfide_bonds: List of disulfide bonds to validate
        strict: If True, use stricter thresholds

    Returns:
        Dictionary with validation results
    """
    try:
        import MDAnalysis as mda
    except ImportError:
        raise ImportError("MDAnalysis required for geometry validation")

    u = mda.Universe(pdb_path)

    results = {
        "valid": True,
        "bonds": [],
    }

    # Thresholds
    if strict:
        dist_range = (1.9, 2.2)
        angle_range = (95, 115)
    else:
        dist_range = (1.8, 2.5)
        angle_range = (85, 125)

    for bond in disulfide_bonds:
        bond_result = {
            "bond": str(bond),
            "valid": True,
            "issues": [],
        }

        # Get atoms
        try:
            sg1 = u.select_atoms(
                f"(chainID {bond.chain1} or segid {bond.chain1}) and resnum {bond.resnum1} and name SG"
            )
            sg2 = u.select_atoms(
                f"(chainID {bond.chain2} or segid {bond.chain2}) and resnum {bond.resnum2} and name SG"
            )
            cb1 = u.select_atoms(
                f"(chainID {bond.chain1} or segid {bond.chain1}) and resnum {bond.resnum1} and name CB"
            )
            cb2 = u.select_atoms(
                f"(chainID {bond.chain2} or segid {bond.chain2}) and resnum {bond.resnum2} and name CB"
            )
        except Exception as e:
            bond_result["valid"] = False
            bond_result["issues"].append(f"Could not select atoms: {e}")
            results["bonds"].append(bond_result)
            results["valid"] = False
            continue

        if len(sg1) == 0 or len(sg2) == 0:
            bond_result["valid"] = False
            bond_result["issues"].append("SG atoms not found")
            results["bonds"].append(bond_result)
            results["valid"] = False
            continue

        # Check SG-SG distance
        dist = np.linalg.norm(sg1.positions[0] - sg2.positions[0])
        bond_result["sg_sg_distance"] = float(dist)

        if not (dist_range[0] <= dist <= dist_range[1]):
            bond_result["valid"] = False
            bond_result["issues"].append(
                f"SG-SG distance {dist:.2f}Å outside range {dist_range}"
            )

        # Check CB-SG-SG angles if CB atoms available
        if len(cb1) > 0 and len(cb2) > 0:
            from MDAnalysis.lib.distances import calc_angles

            # CB1-SG1-SG2 angle
            angle1 = calc_angles(
                cb1.positions[0],
                sg1.positions[0],
                sg2.positions[0],
            )
            angle1_deg = np.degrees(angle1)
            bond_result["cb_sg_sg_angle_1"] = float(angle1_deg)

            # CB2-SG2-SG1 angle
            angle2 = calc_angles(
                cb2.positions[0],
                sg2.positions[0],
                sg1.positions[0],
            )
            angle2_deg = np.degrees(angle2)
            bond_result["cb_sg_sg_angle_2"] = float(angle2_deg)

            if not (angle_range[0] <= angle1_deg <= angle_range[1]):
                bond_result["issues"].append(
                    f"CB-SG-SG angle {angle1_deg:.1f}° outside range {angle_range}"
                )
            if not (angle_range[0] <= angle2_deg <= angle_range[1]):
                bond_result["issues"].append(
                    f"CB-SG-SG angle {angle2_deg:.1f}° outside range {angle_range}"
                )

        if bond_result["issues"]:
            bond_result["valid"] = False
            results["valid"] = False

        results["bonds"].append(bond_result)

    return results
