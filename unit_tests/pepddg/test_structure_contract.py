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


def test_real_1cbw_mutation_mapping() -> None:
    root = Path(__file__).resolve().parents[2]
    structure = root / "research/pepddg_v5/data/independent_validation/scoring_input/1CBW.pdb"
    spec = ComplexSpec(
        structure,
        peptide_chain="I",
        receptor_chains=("F", "G", "H"),
        mutations=(MutationSpec("TI11A", "I", 11, "", "T", "A"),),
    )
    result = validate_complex(spec)
    assert result.mutation_ids == ("TI11A",)
    assert len(result.peptide_residues) == 58


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
