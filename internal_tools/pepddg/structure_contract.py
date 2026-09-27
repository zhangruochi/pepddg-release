"""Strict molecular identity and topology checks for structural PepDDG inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import pandas as pd


_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
_STANDARD = frozenset(_ONE_LETTER.values())
_MUTATION_COLUMNS = ("mutation", "chain", "resnum", "icode", "wt", "mut")


class UnsupportedChemistry(ValueError):
    """The requested chemical graph is outside this validated workflow."""


@dataclass(frozen=True)
class MutationSpec:
    label: str
    chain: str
    resnum: int
    icode: str
    wt: str
    mut: str


@dataclass(frozen=True)
class ComplexSpec:
    structure_path: str | Path
    peptide_chain: str
    receptor_chains: tuple[str, ...]
    mutations: tuple[MutationSpec, ...]
    closure_kind: str = "linear"


@dataclass(frozen=True)
class ComplexValidation:
    structure_sha256: str
    peptide_chain: str
    receptor_chains: tuple[str, ...]
    mutation_ids: tuple[str, ...]
    peptide_residues: tuple[tuple[int, str, str], ...]
    closure_kind: str
    excluded_water_atoms: int


def _icode(value: str) -> str:
    return "" if value in {"", " ", "\x00", ".", "?"} else value


def read_mutations_csv(path: str | Path) -> tuple[MutationSpec, ...]:
    """Read an explicit one-substitution-per-row mutation manifest."""
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [name for name in _MUTATION_COLUMNS if name not in frame]
    if missing:
        raise ValueError("mutation CSV missing columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("mutation CSV is empty")
    mutations: list[MutationSpec] = []
    for index, row in frame.iterrows():
        try:
            numeric = float(row["resnum"])
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("non-integral residue number")
            number = int(numeric)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid resnum at mutation row {index + 1}") from exc
        mutations.append(
            MutationSpec(
                label=str(row["mutation"]).strip(),
                chain=str(row["chain"]).strip(),
                resnum=number,
                icode=_icode(str(row["icode"]).strip()),
                wt=str(row["wt"]).strip().upper(),
                mut=str(row["mut"]).strip().upper(),
            )
        )
    return tuple(mutations)


def validate_complex(spec: ComplexSpec) -> ComplexValidation:
    """Validate structural identity without modifying or silently dropping atoms."""
    try:
        import gemmi
    except ImportError as exc:
        raise RuntimeError("structural validation needs the pepddg[structure] extra (gemmi)") from exc

    path = Path(spec.structure_path)
    if not path.is_file() or path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError("structure must be an existing PDB or mmCIF file")
    if spec.closure_kind != "linear":
        raise UnsupportedChemistry(f"unsupported peptide closure: {spec.closure_kind}")
    if not spec.peptide_chain or not spec.receptor_chains:
        raise ValueError("explicit peptide and receptor chain IDs are required")
    if spec.peptide_chain in spec.receptor_chains or len(set(spec.receptor_chains)) != len(spec.receptor_chains):
        raise ValueError("peptide and receptor chains must be distinct")
    if not spec.mutations:
        raise ValueError("at least one mutation is required")

    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError("one coordinate model is required")
    model = structure[0]
    chains = {chain.name: chain for chain in model}
    selected = (spec.peptide_chain,) + spec.receptor_chains
    missing = [chain for chain in selected if chain not in chains]
    if missing:
        raise ValueError("missing structure chains: " + ", ".join(missing))

    residues: dict[tuple[str, int, str], object] = {}
    excluded_water_atoms = 0
    for chain_name in selected:
        for residue in chains[chain_name]:
            if residue.name in {"HOH", "WAT"} and residue.het_flag != "A":
                excluded_water_atoms += len(residue)
                continue
            if residue.name not in _ONE_LETTER or residue.het_flag != "A":
                raise UnsupportedChemistry(
                    f"nonstandard residue or HETATM in selected chain {chain_name}: {residue.name}"
                )
            key = (chain_name, residue.seqid.num, _icode(residue.seqid.icode))
            if key in residues:
                raise ValueError(f"duplicate residue identity: {key}")
            atoms = list(residue)
            if any(atom.altloc not in {"\x00", " "} for atom in atoms):
                raise UnsupportedChemistry(f"alternate atom locations require preparation: {key}")
            if not {"N", "CA", "C", "O"}.issubset({atom.name for atom in atoms}):
                raise ValueError(f"incomplete backbone at residue {key}")
            residues[key] = residue

    for connection in structure.connections:
        if spec.peptide_chain in {connection.partner1.chain_name, connection.partner2.chain_name}:
            raise UnsupportedChemistry("explicit peptide covalent connections require topology validation")

    peptide = chains[spec.peptide_chain]
    peptide_residues = [residue for residue in peptide if residue.name in _ONE_LETTER]
    sulfurs = [
        (residue.seqid.num, atom)
        for residue in peptide_residues if residue.name == "CYS"
        for atom in residue if atom.name == "SG"
    ]
    for index, (first_number, first) in enumerate(sulfurs):
        for second_number, second in sulfurs[index + 1:]:
            if first_number == second_number:
                continue
            distance = np.linalg.norm(
                np.array([first.pos.x, first.pos.y, first.pos.z])
                - np.array([second.pos.x, second.pos.y, second.pos.z])
            )
            if distance < 2.5:
                raise UnsupportedChemistry("possible peptide disulfide closure detected")
    if len(peptide_residues) >= 2:
        first = next(atom for atom in peptide_residues[0] if atom.name == "N")
        last = next(atom for atom in peptide_residues[-1] if atom.name == "C")
        distance = np.linalg.norm(
            np.array([first.pos.x, first.pos.y, first.pos.z])
            - np.array([last.pos.x, last.pos.y, last.pos.z])
        )
        if distance < 1.9:
            raise UnsupportedChemistry("possible head-to-tail peptide closure detected")

    if any(_icode(residue.seqid.icode) for residue in peptide_residues):
        raise UnsupportedChemistry("peptide insertion codes are unsupported by ProteinMPNN indexing")
    if any(second.seqid.num != first.seqid.num + 1
           for first, second in zip(peptide_residues, peptide_residues[1:])):
        raise UnsupportedChemistry("peptide numbering gap is unsupported by ProteinMPNN indexing")

    labels: set[str] = set()
    identities: set[tuple[str, int, str, str]] = set()
    for mutation in spec.mutations:
        if not mutation.label or mutation.label in labels:
            raise ValueError("duplicate or empty mutation label")
        labels.add(mutation.label)
        if mutation.chain != spec.peptide_chain:
            raise ValueError("mutation must belong to the declared peptide chain")
        if mutation.wt not in _STANDARD or mutation.mut not in _STANDARD or mutation.wt == mutation.mut:
            raise UnsupportedChemistry("one standard-amino-acid substitution is required")
        key = (mutation.chain, mutation.resnum, _icode(mutation.icode))
        residue = residues.get(key)
        if residue is None:
            raise ValueError(f"mutation residue absent or numbering/icode mismatch: {key}")
        observed = _ONE_LETTER[residue.name]
        if observed != mutation.wt:
            raise ValueError(f"wild-type mismatch for {mutation.label}: expected {mutation.wt}, structure {observed}")
        identity = (*key, mutation.mut)
        if identity in identities:
            raise ValueError("duplicate molecular mutation identity")
        identities.add(identity)

    peptide_listing = tuple(
        (residue.seqid.num, _icode(residue.seqid.icode), _ONE_LETTER[residue.name])
        for residue in peptide_residues
    )
    return ComplexValidation(
        structure_sha256=sha256(path.read_bytes()).hexdigest(),
        peptide_chain=spec.peptide_chain,
        receptor_chains=spec.receptor_chains,
        mutation_ids=tuple(m.label for m in spec.mutations),
        peptide_residues=peptide_listing,
        closure_kind=spec.closure_kind,
        excluded_water_atoms=excluded_water_atoms,
    )
