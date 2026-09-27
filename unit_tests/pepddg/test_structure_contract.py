"""Molecular identity checks for new structure inputs."""

from __future__ import annotations

from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from pepddg.structure_contract import (
    ComplexSpec,
    MutationSpec,
    UnsupportedChemistry,
    read_mutations_csv,
    validate_complex,
)


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
