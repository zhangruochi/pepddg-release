"""Real OpenMM topology checks for the supported disulfide closure."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openmm")
pytest.importorskip("pdbfixer")

from openmm import HarmonicBondForce, System
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile

from pepddg._backend.variant_builder import VariantSpec, build_variant_pdb
from pepddg.structure_contract import ComplexSpec, MutationSpec, validate_complex
from pepddg.structural_pipeline import _selected_pdb


@pytest.fixture(scope="module")
def prepared_3otj(tmp_path_factory: pytest.TempPathFactory):
    root = Path(__file__).resolve().parents[2]
    source = root / "research/pepddg_v5/data/independent_validation/scoring_input/3OTJ.pdb"
    spec = ComplexSpec(
        source, "I", ("E",), (MutationSpec("TI11A", "I", 11, "", "T", "A"),), "disulfide"
    )
    expected = validate_complex(spec).disulfide_pairs
    work = tmp_path_factory.mktemp("prepared-3otj-disulfide")
    base = _selected_pdb(spec, work / "selected.pdb")
    wt = Path(build_variant_pdb(str(base), None, str(work / "wt")))
    mutant = Path(build_variant_pdb(
        str(base), VariantSpec("T", "I", 11, "A"), str(work / "mutant")
    ))
    return expected, (wt, mutant)


def test_real_prepared_wt_and_mutant_have_parameterized_disulfides(prepared_3otj) -> None:
    from pepddg.cyclic_integrity import verify_disulfide_integrity

    expected, paths = prepared_3otj
    forcefield = ForceField("amber14-all.xml", "implicit/obc2.xml")
    for path in paths:
        model = PDBFile(str(path))
        system = forcefield.createSystem(model.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        report = verify_disulfide_integrity(
            model.topology, model.positions, peptide_chain="I", expected_pairs=expected,
            system=system, stage=path.parent.name,
        )
        assert set(report) == {"5-55", "14-38", "30-51"}


def test_disulfide_integrity_rejects_unparameterized_bonds(prepared_3otj) -> None:
    from pepddg.cyclic_integrity import verify_disulfide_integrity

    expected, paths = prepared_3otj
    model = PDBFile(str(paths[0]))
    system = System()
    for _ in model.topology.atoms():
        system.addParticle(12.0)
    with pytest.raises(ValueError, match="force-field disulfide bond"):
        verify_disulfide_integrity(
            model.topology, model.positions, peptide_chain="I", expected_pairs=expected,
            system=system, stage="unparameterized",
        )


def test_disulfide_integrity_rejects_extra_receptor_peptide_bond(prepared_3otj) -> None:
    from pepddg.cyclic_integrity import verify_disulfide_integrity

    expected, paths = prepared_3otj
    model = PDBFile(str(paths[0]))
    sulfurs = {
        (atom.residue.chain.id, int(atom.residue.id)): atom
        for atom in model.topology.atoms() if atom.name == "SG"
    }
    model.topology.addBond(sulfurs[("E", 58)], sulfurs[("I", 14)])
    with pytest.raises(ValueError, match="disulfide topology mismatch"):
        verify_disulfide_integrity(
            model.topology, model.positions, peptide_chain="I", expected_pairs=expected,
            stage="cross-chain-bond",
        )


def test_disulfide_integrity_rejects_zero_force_constant(prepared_3otj) -> None:
    from pepddg.cyclic_integrity import verify_disulfide_integrity

    expected, paths = prepared_3otj
    model = PDBFile(str(paths[0]))
    system = ForceField("amber14-all.xml", "implicit/obc2.xml").createSystem(
        model.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
    sulfurs = {
        (atom.residue.chain.id, int(atom.residue.id)): atom.index
        for atom in model.topology.atoms() if atom.name == "SG"
    }
    pair = {sulfurs[("I", 5)], sulfurs[("I", 55)]}
    for force in system.getForces():
        if isinstance(force, HarmonicBondForce):
            for index in range(force.getNumBonds()):
                first, second, length, stiffness = force.getBondParameters(index)
                if {int(first), int(second)} == pair:
                    force.setBondParameters(index, first, second, length, 0 * stiffness)
    with pytest.raises(ValueError, match="force-field disulfide bond"):
        verify_disulfide_integrity(
            model.topology, model.positions, peptide_chain="I", expected_pairs=expected,
            system=system, stage="zero-stiffness",
        )
