"""Regression tests for new-structure orchestration and paired restart semantics."""

from pathlib import Path
from dataclasses import replace
import subprocess
import sys

import pytest

from pepddg.structure_contract import ComplexSpec, MutationSpec, UnsupportedChemistry
from pepddg.structural_pipeline import paired_restart_ddg, run_structural_cohort


def _atom(serial: int, name: str, residue: str, chain: str, number: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:3s} {chain}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {name[0]:>2s}\n"
    )


def _spec(tmp_path: Path) -> ComplexSpec:
    lines = []
    serial = 1
    for residue, chain, number, offset in (("ALA", "A", 1, 0.0), ("GLY", "B", 5, 8.0)):
        for name, delta in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
            lines.append(_atom(serial, name, residue, chain, number, offset + delta))
            serial += 1
    path = tmp_path / "complex.pdb"
    path.write_text("".join(lines) + "END\n")
    return ComplexSpec(path, "B", ("A",), (MutationSpec("G5A", "B", 5, "", "G", "A"),))


def test_paired_restarts_require_three_matching_finite_indices() -> None:
    assert paired_restart_ddg([2.0, 4.0, 10.0], [3.0, 7.0, 9.0]) == 1.0
    with pytest.raises(ValueError, match="three"):
        paired_restart_ddg([2.0, float("nan"), 10.0], [3.0, 7.0, 9.0])
    with pytest.raises(ValueError, match="equal"):
        paired_restart_ddg([2.0, 4.0, 10.0], [3.0, 7.0])


def test_structural_cohort_uses_raw_channels_and_records_provenance(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    calls = []

    def build(base, variant, output):
        calls.append(("build", variant))
        return str(base)

    def physics(pdb_path, receptor_chain, ligand_chain, **kwargs):
        calls.append(("physics", tuple(kwargs["restraint_exclusion_residues"])))
        offset = 0.0 if len([x for x in calls if x[0] == "physics"]) == 1 else 2.0
        return {
            "dg_bind_kcal_mol_restarts": [offset, offset + 1, offset + 2],
            "e_cross_interface_total_screened_kcal_mol_restarts": [offset + 1, offset + 2, offset + 3],
        }

    monkeypatch.setattr(pipeline, "_build_variant_pdb", build)
    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4)})
    result = run_structural_cohort(_spec(tmp_path), target="T", parent_id="WT", output_dir=tmp_path / "out", n_restarts=3)
    row = result.features.iloc[0]
    assert row["ddg_bind_proxy"] == 2.0
    assert row["ddg_xint_iface"] == 2.0
    assert row["mpnn_neg_llr_complex"] == 0.2
    assert row["mpnn_ddg_bind"] == 0.4
    assert calls[1] == ("physics", ("B:5",))
    assert calls[3] == ("physics", ("B:5",))
    assert result.provenance["n_restarts"] == 3
    assert (tmp_path / "out" / "features.csv").exists()
    assert (tmp_path / "out" / "scores.csv").exists()


def test_cyclic_input_fails_before_any_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    cyclic = ComplexSpec(spec.structure_path, spec.peptide_chain, spec.receptor_chains, spec.mutations, "head_to_tail")
    with pytest.raises(UnsupportedChemistry):
        run_structural_cohort(cyclic, target="T", parent_id="WT", output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_structure_cli_rejects_unsupported_chemistry_without_output(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    mutations = tmp_path / "mutations.csv"
    mutations.write_text("mutation,chain,resnum,icode,wt,mut\nG5A,B,5,,G,A\n")
    destination = tmp_path / "results"
    process = subprocess.run(
        [sys.executable, "-m", "pepddg", "score-structures", "--structure", str(spec.structure_path),
         "--peptide-chain", "B", "--receptor-chain", "A", "--mutations", str(mutations),
         "--target", "T", "--parent-id", "WT", "--closure", "head_to_tail", "--output", str(destination)],
        capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "unsupported peptide closure" in process.stderr
    assert not destination.exists()


def test_mmcif_conversion_keeps_selected_chains(tmp_path: Path) -> None:
    import gemmi
    from pepddg.structural_pipeline import _selected_pdb

    spec = _spec(tmp_path)
    cif = tmp_path / "complex.cif"
    gemmi.read_structure(str(spec.structure_path)).make_mmcif_document().write_file(str(cif))
    converted = _selected_pdb(ComplexSpec(cif, "B", ("A",), spec.mutations), tmp_path / "selected.pdb")
    observed = gemmi.read_structure(str(converted))
    assert {chain.name for chain in observed[0]} == {"A", "B"}
    assert len(observed[0]["A"]) == len(observed[0]["B"]) == 1


def test_interrupted_cohort_resumes_completed_mutations(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    first = _spec(tmp_path)
    second = MutationSpec("G5V", "B", 5, "", "G", "V")
    spec = replace(first, mutations=(first.mutations[0], second))
    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4), "G5V": (0.3, 0.5)})
    calls = []

    def physics(pdb_path, receptor_chain, ligand_chain, **kwargs):
        calls.append(len(calls))
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        offset = float(len(calls))
        return {
            "dg_bind_kcal_mol_restarts": [offset] * 3,
            "e_cross_interface_total_screened_kcal_mol_restarts": [offset] * 3,
        }

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="interruption"):
        run_structural_cohort(spec, target="T", parent_id="WT", output_dir=output, n_restarts=3)
    assert not (output / "scores.csv").exists()
    assert (output / ".pepddg-work" / "mutations" / "0000.json").exists()

    result = run_structural_cohort(spec, target="T", parent_id="WT", output_dir=output, n_restarts=3)
    assert result.features["mutation"].tolist() == ["G5A", "G5V"]
    assert len(calls) == 5
    assert (output / "scores.csv").exists()


def test_changed_input_refuses_saved_partial_work(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    spec = _spec(tmp_path)
    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4)})

    def physics(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="interruption"):
        run_structural_cohort(spec, target="T", parent_id="WT", output_dir=output, n_restarts=3)
    with pytest.raises(ValueError, match="checkpoint identity"):
        run_structural_cohort(spec, target="T", parent_id="WT", output_dir=output, n_restarts=4)


def test_each_mutant_uses_its_own_paired_wt_protocol(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    first = _spec(tmp_path)
    pdb = first.structure_path
    addition = "".join(_atom(20 + index, name, "ALA", "B", 6, 12.0 + delta)
                       for index, (name, delta) in enumerate((("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0))))
    pdb.write_text(pdb.read_text().replace("END\n", addition + "END\n"))
    spec = replace(first, mutations=(first.mutations[0], MutationSpec("A6V", "B", 6, "", "A", "V")))
    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4), "A6V": (0.3, 0.5)})
    observed = []

    def physics(*args, **kwargs):
        observed.append(tuple(kwargs["restraint_exclusion_residues"]))
        offset = float(len(observed))
        return {
            "dg_bind_kcal_mol_restarts": [offset] * 3,
            "e_cross_interface_total_screened_kcal_mol_restarts": [offset] * 3,
        }

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    run_structural_cohort(spec, target="T", parent_id="WT", output_dir=tmp_path / "out", n_restarts=3)
    assert observed == [("B:5",), ("B:5",), ("B:6",), ("B:6",)]


def test_target_union_wt_scope_scores_wt_once_and_records_protocol(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    first = _spec(tmp_path)
    pdb = first.structure_path
    addition = "".join(_atom(20 + index, name, "ALA", "B", 6, 12.0 + delta)
                       for index, (name, delta) in enumerate((("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0))))
    pdb.write_text(pdb.read_text().replace("END\n", addition + "END\n"))
    spec = replace(first, mutations=(
        first.mutations[0], MutationSpec("G5V", "B", 5, "", "G", "V"),
        MutationSpec("A6V", "B", 6, "", "A", "V"),
    ))
    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {
        "G5A": (0.2, 0.4), "G5V": (0.3, 0.5), "A6V": (0.4, 0.6)
    })
    observed = []

    def physics(*args, **kwargs):
        observed.append(tuple(kwargs["restraint_exclusion_residues"]))
        value = float(len(observed))
        return {
            "dg_bind_kcal_mol_restarts": [value] * 3,
            "e_cross_interface_total_screened_kcal_mol_restarts": [value] * 3,
        }

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    result = run_structural_cohort(
        spec, target="T", parent_id="WT", output_dir=tmp_path / "union",
        n_restarts=3, wt_restraint_scope="target_union",
    )
    assert observed == [("B:5", "B:6"), ("B:5",), ("B:5",), ("B:6",)]
    assert result.features["ddg_bind_proxy"].tolist() == [1.0, 2.0, 3.0]
    assert result.provenance["wt_restraint_scope"] == "target_union"


def test_target_union_resume_reuses_wt_and_refuses_scope_change(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    first = _spec(tmp_path)
    spec = replace(first, mutations=(first.mutations[0], MutationSpec("G5V", "B", 5, "", "G", "V")))
    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4), "G5V": (0.3, 0.5)})
    calls = []

    def physics(*args, **kwargs):
        calls.append(tuple(kwargs["restraint_exclusion_residues"]))
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        value = float(len(calls))
        return {
            "dg_bind_kcal_mol_restarts": [value] * 3,
            "e_cross_interface_total_screened_kcal_mol_restarts": [value] * 3,
        }

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    output = tmp_path / "union-interrupted"
    with pytest.raises(RuntimeError, match="interruption"):
        run_structural_cohort(
            spec, target="T", parent_id="WT", output_dir=output,
            n_restarts=3, wt_restraint_scope="target_union",
        )
    with pytest.raises(ValueError, match="checkpoint identity"):
        run_structural_cohort(spec, target="T", parent_id="WT", output_dir=output, n_restarts=3)
    result = run_structural_cohort(
        spec, target="T", parent_id="WT", output_dir=output,
        n_restarts=3, wt_restraint_scope="target_union",
    )
    assert result.features["mutation"].tolist() == ["G5A", "G5V"]
    assert len(calls) == 4
    assert (output / "scores.csv").exists()


def test_target_union_accepts_original_pool_sites_beyond_scored_rows(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    spec = _spec(tmp_path)
    pdb = spec.structure_path
    addition = "".join(_atom(20 + index, name, "ALA", "B", 6, 12.0 + delta)
                       for index, (name, delta) in enumerate((("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0))))
    pdb.write_text(pdb.read_text().replace("END\n", addition + "END\n"))
    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4)})
    observed = []

    def physics(*args, **kwargs):
        observed.append(tuple(kwargs["restraint_exclusion_residues"]))
        value = float(len(observed))
        return {
            "dg_bind_kcal_mol_restarts": [value] * 3,
            "e_cross_interface_total_screened_kcal_mol_restarts": [value] * 3,
        }

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    result = run_structural_cohort(
        spec, target="T", parent_id="WT", output_dir=tmp_path / "original-pool",
        n_restarts=3, wt_restraint_scope="target_union", wt_union_sites=("B:6", "B:5"),
    )
    assert observed == [("B:6", "B:5"), ("B:5",)]
    assert result.provenance["wt_union_sites"] == ["B:6", "B:5"]


def test_target_union_rejects_incomplete_or_invalid_site_pools(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    for sites, message in (((), "omits"), (("A:5", "B:5"), "invalid")):
        output = tmp_path / f"rejected-{message}"
        with pytest.raises(ValueError, match=message):
            run_structural_cohort(
                spec, target="T", parent_id="WT", output_dir=output,
                n_restarts=3, wt_restraint_scope="target_union", wt_union_sites=sites,
            )
        assert not output.exists()


def test_final_publication_interruption_resumes_only_own_outputs(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_mpnn", lambda *a, **k: {"G5A": (0.2, 0.4)})
    calls = []

    def physics(*args, **kwargs):
        calls.append(1)
        value = float(len(calls))
        return {
            "dg_bind_kcal_mol_restarts": [value] * 3,
            "e_cross_interface_total_screened_kcal_mol_restarts": [value] * 3,
        }

    monkeypatch.setattr(pipeline, "_score_openmm", physics)
    original_replace = pipeline.os.replace

    def interrupted_replace(source, destination):
        if Path(destination).name == "scores.csv":
            raise OSError("publication interrupted")
        return original_replace(source, destination)

    output = tmp_path / "out"
    monkeypatch.setattr(pipeline.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="publication interrupted"):
        run_structural_cohort(_spec(tmp_path), target="T", parent_id="WT", output_dir=output, n_restarts=3)
    assert (output / "features.csv").exists()
    monkeypatch.setattr(pipeline.os, "replace", original_replace)
    result = run_structural_cohort(_spec(tmp_path), target="T", parent_id="WT", output_dir=output, n_restarts=3)
    assert result.features["mutation"].tolist() == ["G5A"]
    assert len(calls) == 2
    assert (output / "scores.csv").exists()


def test_changed_producer_code_refuses_saved_partial_work(tmp_path: Path, monkeypatch) -> None:
    import pepddg.structural_pipeline as pipeline

    monkeypatch.setattr(pipeline, "_build_variant_pdb", lambda base, variant, output: str(base))
    monkeypatch.setattr(pipeline, "_score_openmm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted")))
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_structural_cohort(_spec(tmp_path), target="T", parent_id="WT", output_dir=output, n_restarts=3)
    original_read = Path.read_bytes

    def changed_producer(path):
        content = original_read(path)
        if path.name == "structural_features.py":
            return content + b"# simulated producer change\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_producer)
    with pytest.raises(ValueError, match="checkpoint identity"):
        run_structural_cohort(_spec(tmp_path), target="T", parent_id="WT", output_dir=output, n_restarts=3)
