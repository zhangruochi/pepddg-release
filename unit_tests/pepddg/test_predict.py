"""Public WT-structure API: identity, single-candidate semantics and delegation."""
from pathlib import Path

import pandas as pd
import pytest

from pepddg import predict
from pepddg.api import score_features, FEATURE_COLUMNS, IDENTITY_COLUMNS
from pepddg.structural_pipeline import StructuralResult
from pepddg.structure_contract import UnsupportedChemistry

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT / "examples/skempi_cyclic/data/1SMF/complex.pdb"


def _fake_runner(calls):
    def run(spec, **kwargs):
        calls.append((spec, kwargs))
        rows = [{"target": kwargs["target"], "parent_id": kwargs["parent_id"],
                 "mutation": m.label, "ddg_xint_iface": float(i - 1),
                 "ddg_bind_proxy": float(i + 1), "n_iface_contacts_8a": 5 + i,
                 "n_neighbors_10a": 3 + i, "mpnn_neg_llr_complex": .2 + i,
                 "mpnn_ddg_bind": .4 + i} for i, m in enumerate(spec.mutations)]
        features = pd.DataFrame(rows)
        return StructuralResult(features, score_features(features), {"structure_sha256": "fixture"})
    return run


def test_single_mutation_keeps_channels_without_trivial_zero_rank(monkeypatch, tmp_path):
    import pepddg.prediction as module
    calls = []
    monkeypatch.setattr(module, "run_structural_cohort", _fake_runner(calls))
    result = predict(WT, "T2A", peptide_chain="I", receptor_chain="E", output_dir=tmp_path / "one")
    assert isinstance(result, pd.DataFrame)
    assert result.mutation.tolist() == ["T2A"]
    assert result.ddg_xint_iface.tolist() == [-1.0]
    assert result.rankscore_pepddg_zs.isna().all()
    assert result.attrs["rank_available"] is False
    assert calls[0][0].closure_kind == "disulfide"
    assert calls[0][0].mutations[0].resnum == 2
    assert calls[0][1]["n_restarts"] == 7


def test_mutation_series_is_one_cohort_and_preserves_input_order(monkeypatch, tmp_path):
    import pepddg.prediction as module
    calls = []
    monkeypatch.setattr(module, "run_structural_cohort", _fake_runner(calls))
    result = predict(WT, ["T2A", "K3A", "S4A"], peptide_chain="I", receptor_chain="E",
                     output_dir=tmp_path / "batch", platform="CUDA", seed=27)
    assert result.mutation.tolist() == ["T2A", "K3A", "S4A"]
    assert result.rankscore_pepddg_zs.notna().all()
    assert result.attrs["rank_available"] is True
    assert len(calls) == 1
    assert calls[0][1]["platform"] == "CUDA"
    assert calls[0][1]["seed"] == 27
    expected = score_features(result[list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)])
    pd.testing.assert_frame_equal(result, expected.table)


@pytest.mark.parametrize("mutations", [[], ["T2A", "T02A"], "T2T", "T2A+K3A", [2], {"T2A"}, "A2V"])
def test_bad_mutations_fail_before_execution_or_output(monkeypatch, tmp_path, mutations):
    import pepddg.prediction as module
    calls = []
    monkeypatch.setattr(module, "run_structural_cohort", _fake_runner(calls))
    output = tmp_path / "bad"
    with pytest.raises((ValueError, TypeError)):
        predict(WT, mutations, peptide_chain="I", receptor_chain="E", output_dir=output)
    assert not calls
    assert not output.exists()


def test_explicit_linear_does_not_bypass_disulfide_guard(monkeypatch, tmp_path):
    import pepddg.prediction as module
    calls = []
    monkeypatch.setattr(module, "run_structural_cohort", _fake_runner(calls))
    with pytest.raises(UnsupportedChemistry, match="disulfide"):
        predict(WT, "T2A", peptide_chain="I", receptor_chain="E", closure="linear", output_dir=tmp_path / "bad")
    assert not calls


def test_auto_does_not_swallow_unrelated_unsupported_chemistry(monkeypatch, tmp_path):
    import pepddg.prediction as module
    def refuse(spec):
        raise UnsupportedChemistry("unsupported explicit peptide covalent connection")
    monkeypatch.setattr(module, "validate_complex", refuse)
    with pytest.raises(UnsupportedChemistry, match="covalent"):
        predict(WT, "T2A", peptide_chain="I", receptor_chain="E", output_dir=tmp_path / "bad")


def test_implicit_output_is_retained_and_reported(monkeypatch, tmp_path):
    import pepddg.prediction as module
    output = tmp_path / "auto"
    def allocate(**kwargs):
        output.mkdir()
        return str(output)
    calls = []
    monkeypatch.setattr(module.tempfile, "mkdtemp", allocate)
    monkeypatch.setattr(module, "run_structural_cohort", _fake_runner(calls))
    result = predict(WT, "T2A", peptide_chain="I", receptor_chain="E")
    assert Path(result.attrs["output_dir"]) == output
    assert output.exists()


def test_failure_preserves_checkpoint_location(monkeypatch, tmp_path):
    import pepddg.prediction as module
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(module, "run_structural_cohort", fail)
    output = tmp_path / "resume"
    with pytest.raises(RuntimeError) as info:
        predict(WT, "T2A", peptide_chain="I", receptor_chain="E", output_dir=output)
    assert any(str(output) in note for note in info.value.__notes__)
