"""Preserve chemical terminal caps rather than score a deprotected peptide."""
from pathlib import Path

import pytest

from pepddg.structure_contract import ComplexSpec, MutationSpec, UnsupportedChemistry, validate_complex
from pepddg.structural_pipeline import _selected_pdb


def _line(serial, name, residue, number, x, y=0, z=0, chain="B", het=False):
    element = name[0]
    return (f"{'HETATM' if het else 'ATOM  '}{serial:5d} {name:^4s} {residue:3s} {chain}{number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{20:6.2f}          {element:>2s}\n")


def _capped(tmp_path):
    lines = [_line(i+1, n, "GLY", 1, x,y,chain="A") for i, (n,x,y) in enumerate((("N",10,0),("CA",11.4,0),("C",12,1.3),("O",11.8,2.5)))]
    lines += [_line(i+5,n,"ALA",1,x,y,z) for i,(n,x,y,z) in enumerate((("N",0,0,0),("CA",1.4,0,0),("C",2,1.3,0),("O",1.8,2.5,0),("CB",1.7,-.8,1.1)))]
    lines += ["TER\n", _line(10,"C","ACE",0,-1.33,het=True), _line(11,"O","ACE",0,-1.8,1.1,het=True), _line(12,"CH3","ACE",0,-2.1,-1.2,het=True), _line(13,"N","NH2",2,3.3,1.5,het=True)]
    path = tmp_path / "capped.pdb"
    path.write_text("".join(lines)+"END\n")
    return ComplexSpec(path,"B",("A",),(MutationSpec("A1V","B",1,"","A","V"),))


def test_contract_and_selection_preserve_terminal_caps_and_aa_numbering(tmp_path):
    import gemmi
    spec = _capped(tmp_path)
    validated = validate_complex(spec)
    assert validated.terminal_caps == (("ACE",0),("NH2",2))
    assert validated.peptide_residues == ((1,"","A"),)
    selected = _selected_pdb(spec,tmp_path/"selected.pdb")
    peptide = gemmi.read_structure(str(selected))[0]["B"]
    assert [(r.name,r.seqid.num) for r in peptide] == [("ACE",0),("ALA",1),("NH2",2)]
    assert {a.name for a in peptide[0]} == {"C","O","CH3"}


@pytest.mark.parametrize("old,new",[("NH2","NME"),(" CH3","  CA"),("B   2","B   3")])
def test_unknown_or_malformed_caps_fail_closed(tmp_path,old,new):
    spec = _capped(tmp_path)
    Path(spec.structure_path).write_text(Path(spec.structure_path).read_text().replace(old,new))
    with pytest.raises(UnsupportedChemistry):
        validate_complex(spec)


def test_cap_bond_must_be_covalent_distance(tmp_path):
    spec = _capped(tmp_path)
    text = Path(spec.structure_path).read_text().replace("   3.300", "  30.300")
    Path(spec.structure_path).write_text(text)
    with pytest.raises(UnsupportedChemistry,match="cap"):
        validate_complex(spec)


def test_amber_caps_have_actual_external_bonds_and_forcefield_terms(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    from openmm.app import PDBFile, ForceField, NoCutoff, HBonds
    from pepddg._backend.variant_builder import build_variant_pdb
    from pepddg.terminal_caps import verify_terminal_cap_integrity
    spec = _capped(tmp_path)
    base = _selected_pdb(spec,tmp_path/"selected.pdb")
    # This parameterization fixture is the capped peptide alone: an isolated
    # one-residue free GLY receptor has no Amber template for both termini.
    base.write_text("".join(line for line in base.read_text().splitlines(keepends=True)
                           if line.startswith(("ATOM  ","HETATM")) and line[21] == "B")+"END\n")
    prepared = build_variant_pdb(str(base),None,str(tmp_path/"wt"),preserve_terminal_caps=True)
    model = PDBFile(prepared)
    ff = ForceField("amber14-all.xml","implicit/obc2.xml")
    system = ff.createSystem(model.topology,nonbondedMethod=NoCutoff,constraints=HBonds)
    report = verify_terminal_cap_integrity(model.topology,model.positions,peptide_chain="B",expected_caps=(("ACE",0),("NH2",2)),system=system,stage="test")
    assert set(report) == {"ACE:0","NH2:2"}
    cap = next(r for r in model.topology.residues() if r.name == "NH2")
    assert {a.name for a in cap.atoms()} == {"N","HN1","HN2"}
    # Broken connectivity is a chemical failure even if the cap atoms remain.
    model.topology._bonds = [b for b in model.topology.bonds() if not (b[0].residue != b[1].residue and cap in (b[0].residue,b[1].residue))]
    with pytest.raises(ValueError,match="cap"):
        verify_terminal_cap_integrity(model.topology,model.positions,peptide_chain="B",expected_caps=(("ACE",0),("NH2",2)),stage="broken")


def test_native_preparation_seed_restores_state_even_on_failure():
    import random
    from pepddg.terminal_caps import scoped_preparation_seed
    state = random.getstate()
    with scoped_preparation_seed(123):
        first = random.random()
    assert random.getstate() == state
    with pytest.raises(RuntimeError):
        with scoped_preparation_seed(123):
            assert random.random() == first
            raise RuntimeError("interrupted")
    assert random.getstate() == state


def test_capped_cohort_rejects_missing_cap_qc_before_publication(tmp_path,monkeypatch):
    import pepddg.structural_pipeline as pipeline
    spec = _capped(tmp_path)
    monkeypatch.setattr(pipeline,"_build_variant_pdb",lambda base,*a,**k:str(base))
    monkeypatch.setattr(pipeline,"_score_mpnn",lambda *a,**k:{"A1V":(.2,.4)})
    monkeypatch.setattr(pipeline,"_score_openmm",lambda *a,**k:{
        "dg_bind_kcal_mol_restarts":[1.,2.,3.],
        "e_cross_interface_total_screened_kcal_mol_restarts":[1.,2.,3.],
    })
    with pytest.raises(ValueError,match="terminal cap integrity"):
        pipeline.run_structural_cohort(spec,target="T",parent_id="WT",output_dir=tmp_path/"out",n_restarts=3)
    assert not (tmp_path/"out/scores.csv").exists()
