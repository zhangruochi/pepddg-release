"""Molecular identity checks for new structure inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pepddg.structure_contract import (
    ComplexSpec,
    MutationSpec,
    UnsupportedChemistry,
    read_mutations_csv,
    validate_complex,
)

gemmi = pytest.importorskip("gemmi")


def _atom(serial: int, name: str, residue: str, chain: str, number: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:3s} {chain}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {name[0]:>2s}\n"
    )


def _pdb(path: Path) -> Path:
    lines = []
    serial = 1
    for residue, chain, number, offset in (("ALA", "A", 1, 0.0), ("GLY", "B", 5, 8.0)):
        for name, delta in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            lines.append(_atom(serial, name, residue, chain, number, offset + delta))
            serial += 1
    path.write_text("".join(lines) + "END\n")
    return path


def _spec(path: Path, **mut_changes) -> ComplexSpec:
    fields = dict(label="G5A", chain="B", resnum=5, icode="", wt="G", mut="A")
    fields.update(mut_changes)
    return ComplexSpec(path, peptide_chain="B", receptor_chains=("A",), mutations=(MutationSpec(**fields),))


def test_valid_pdb_and_cif_preserve_mutation_identity(tmp_path: Path) -> None:
    pdb = _pdb(tmp_path / "complex.pdb")
    assert validate_complex(_spec(pdb)).mutation_ids == ("G5A",)
    cif = tmp_path / "complex.cif"
    gemmi.read_structure(str(pdb)).make_mmcif_document().write_file(str(cif))
    assert validate_complex(_spec(cif)).mutation_ids == ("G5A",)


def test_wrong_wild_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wild-type"):
        validate_complex(_spec(_pdb(tmp_path / "complex.pdb"), wt="A", mut="V"))


def test_peptide_numbering_gap_is_rejected_before_mpnn_indexing(tmp_path: Path) -> None:
    pdb = _pdb(tmp_path / "complex.pdb")
    lines = [_atom(20 + index, name, "ALA", "B", 7, 16.0 + delta)
             for index, (name, delta) in enumerate((("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)))]
    pdb.write_text(pdb.read_text().replace("END\n", "".join(lines) + "END\n"))
    spec = _spec(pdb, label="A7V", resnum=7, wt="A", mut="V")
    with pytest.raises(UnsupportedChemistry, match="numbering gap"):
        validate_complex(spec)


def test_cyclic_topology_is_not_silently_treated_as_linear(tmp_path: Path) -> None:
    spec = _spec(_pdb(tmp_path / "complex.pdb"))
    spec = ComplexSpec(spec.structure_path, spec.peptide_chain, spec.receptor_chains, spec.mutations, closure_kind="head_to_tail")
    with pytest.raises(UnsupportedChemistry, match="head_to_tail"):
        validate_complex(spec)


def test_duplicate_mutation_identity_is_rejected(tmp_path: Path) -> None:
    spec = _spec(_pdb(tmp_path / "complex.pdb"))
    spec = ComplexSpec(spec.structure_path, spec.peptide_chain, spec.receptor_chains, spec.mutations * 2)
    with pytest.raises(ValueError, match="duplicate"):
        validate_complex(spec)


def test_real_1cbw_disulfide_is_not_claimed_linear() -> None:
    root = Path(__file__).resolve().parents[2]
    structure = root / "research/pepddg_v5/data/independent_validation/scoring_input/1CBW.pdb"
    spec = ComplexSpec(
        structure,
        peptide_chain="I",
        receptor_chains=("F", "G", "H"),
        mutations=(MutationSpec("TI11A", "I", 11, "", "T", "A"),),
    )
    with pytest.raises(UnsupportedChemistry, match="disulfide"):
        validate_complex(spec)


def test_real_3otj_disulfide_closure_is_identified() -> None:
    root = Path(__file__).resolve().parents[2]
    structure = root / "research/pepddg_v5/data/independent_validation/scoring_input/3OTJ.pdb"
    spec = ComplexSpec(
        structure,
        peptide_chain="I",
        receptor_chains=("E",),
        mutations=(MutationSpec("TI11A", "I", 11, "", "T", "A"),),
        closure_kind="disulfide",
    )
    result = validate_complex(spec)
    assert result.mutation_ids == ("TI11A",)
    assert result.disulfide_pairs == ((5, 55), (14, 38), (30, 51))


def test_disulfide_closure_rejects_mutation_of_bonded_cysteine() -> None:
    root = Path(__file__).resolve().parents[2]
    structure = root / "research/pepddg_v5/data/independent_validation/scoring_input/3OTJ.pdb"
    spec = ComplexSpec(
        structure,
        peptide_chain="I",
        receptor_chains=("E",),
        mutations=(MutationSpec("CI5A", "I", 5, "", "C", "A"),),
        closure_kind="disulfide",
    )
    with pytest.raises(UnsupportedChemistry, match="closure-forming cysteine"):
        validate_complex(spec)


def test_disulfide_closure_rejects_inverted_peptide_stereochemistry(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "research/pepddg_v5/data/independent_validation/scoring_input/3OTJ.pdb"
    structure = gemmi.read_structure(str(source))
    residue = next(residue for residue in structure[0]["I"] if residue.seqid.num == 11)
    atoms = {atom.name: atom for atom in residue}
    def point(atom):
        return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    ca = point(atoms["CA"])
    normal = np.cross(point(atoms["N"]) - ca, point(atoms["C"]) - ca)
    cb = point(atoms["CB"])
    reflected = cb - 2 * np.dot(cb - ca, normal) / np.dot(normal, normal) * normal
    atoms["CB"].pos = gemmi.Position(*reflected)
    path = tmp_path / "inverted.pdb"
    structure.write_pdb(str(path))
    spec = ComplexSpec(
        path, "I", ("E",), (MutationSpec("TI11A", "I", 11, "", "T", "A"),), "disulfide"
    )
    with pytest.raises(UnsupportedChemistry, match="inverted peptide stereochemistry"):
        validate_complex(spec)


def test_disulfide_record_must_match_coordinate_pairing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "research/pepddg_v5/data/independent_validation/scoring_input/3OTJ.pdb"
    path = tmp_path / "wrong-ssbond.pdb"
    path.write_text("SSBOND   1 CYS I    5    CYS I   38\n" + source.read_text())
    spec = ComplexSpec(
        path, "I", ("E",), (MutationSpec("TI11A", "I", 11, "", "T", "A"),), "disulfide"
    )
    with pytest.raises(UnsupportedChemistry, match="disulfide record disagrees"):
        validate_complex(spec)


def test_disulfide_closure_rejects_cross_chain_sulfur_contact(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "research/pepddg_v5/data/independent_validation/scoring_input/3OTJ.pdb"
    structure = gemmi.read_structure(str(source))
    peptide = next(residue for residue in structure[0]["I"] if residue.seqid.num == 14)
    receptor = next(residue for residue in structure[0]["E"] if residue.seqid.num == 58)
    peptide_sg = next(atom for atom in peptide if atom.name == "SG")
    receptor_sg = next(atom for atom in receptor if atom.name == "SG")
    receptor_sg.pos = gemmi.Position(peptide_sg.pos.x + 2.0, peptide_sg.pos.y, peptide_sg.pos.z)
    path = tmp_path / "cross-chain-sulfur.pdb"
    structure.write_pdb(str(path))
    spec = ComplexSpec(
        path, "I", ("E",), (MutationSpec("TI11A", "I", 11, "", "T", "A"),), "disulfide"
    )
    with pytest.raises(UnsupportedChemistry, match="cross-chain sulfur contact"):
        validate_complex(spec)


def test_fractional_residue_number_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "mutations.csv"
    source.write_text("mutation,chain,resnum,icode,wt,mut\nG5A,B,5.5,,G,A\n")
    with pytest.raises(ValueError, match="resnum"):
        read_mutations_csv(source)


def test_ambiguous_alternate_location_is_rejected(tmp_path: Path) -> None:
    pdb = _pdb(tmp_path / "complex.pdb")
    text = pdb.read_text().replace(" CA  GLY B", " CA AGLY B")
    pdb.write_text(text)
    with pytest.raises(UnsupportedChemistry, match="alternate"):
        validate_complex(_spec(pdb))


def test_peptide_disulfide_closure_is_not_scored_as_linear(tmp_path: Path) -> None:
    pdb = tmp_path / "disulfide.pdb"
    lines = []
    serial = 1
    for residue, chain, number, offset in (("ALA", "A", 1, 0.0), ("CYS", "B", 5, 8.0), ("CYS", "B", 6, 16.0)):
        for name, delta in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            lines.append(_atom(serial, name, residue, chain, number, offset + delta))
            serial += 1
        if chain == "B":
            lines.append(_atom(serial, "SG", residue, chain, number, 12.0 if number == 5 else 14.0))
            serial += 1
    pdb.write_text("".join(lines) + "END\n")
    with pytest.raises(UnsupportedChemistry, match="disulfide"):
        validate_complex(_spec(pdb, wt="C", mut="A"))


def test_crystal_waters_are_reported_and_excluded_from_polymer_validation(tmp_path: Path) -> None:
    pdb = _pdb(tmp_path / "complex.pdb")
    pdb.write_text(pdb.read_text().replace("END\n", "HETATM   90  O   HOH B  10      25.000   0.000   0.000  1.00 20.00           O\nEND\n"))
    result = validate_complex(_spec(pdb))
    assert result.excluded_water_atoms == 1
