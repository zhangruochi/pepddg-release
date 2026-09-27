"""Public scoring must not turn incomplete feature tables into rankings."""

from __future__ import annotations

import pandas as pd
import pytest
import subprocess
import sys

from pepddg import score_features


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["target1", "target1"],
            "parent_id": ["WT1", "WT1"],
            "mutation": ["A1V", "A2L"],
            "ddg_xint_iface": [0.1, 1.2],
            "ddg_bind_proxy": [0.2, 1.1],
            "n_iface_contacts_8a": [8, 4],
            "n_neighbors_10a": [12, 7],
            "mpnn_neg_llr_complex": [0.3, 0.8],
            "mpnn_ddg_bind": [0.2, 0.9],
        }
    )


def test_scores_complete_raw_feature_cohort_without_changing_row_identity() -> None:
    result = score_features(_features())
    assert result.table["mutation"].tolist() == ["A1V", "A2L"]
    assert result.table["rankscore_pepddg_zs"].tolist() == [0.0, 1.0]
    assert result.summary["n_rows"] == 2
    assert result.summary["score_kind"] == "relative_cohort_rank"
    assert result.summary["lower_is_better"] is True


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda x: x.drop(columns="mpnn_ddg_bind"), "mpnn_ddg_bind"),
        (lambda x: x.assign(mpnn_ddg_bind=[0.2, float("nan")]), "finite"),
        (lambda x: x.assign(mutation=["A1V", "A1V"]), "duplicate"),
        (lambda x: x.assign(target=["target1", "target2"]), "one target"),
        (lambda x: x.assign(ddg_exp=[-0.5, 1.2]), "experimental"),
        (lambda x: x.assign(rankscore_3view_base=[0.0, 1.0]), "precomputed"),
    ],
)
def test_rejects_incomplete_or_leaking_inputs(change, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        score_features(change(_features()))


def test_cli_returns_failure_without_scored_output(tmp_path) -> None:
    source = tmp_path / "bad.csv"
    destination = tmp_path / "scores.csv"
    _features().drop(columns="mpnn_ddg_bind").to_csv(source, index=False)
    process = subprocess.run(
        [sys.executable, "-m", "pepddg", "score-features", "--input", str(source), "--output", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "mpnn_ddg_bind" in process.stderr
    assert not destination.exists()


def test_cli_outputs_scores_and_provenance(tmp_path) -> None:
    source = tmp_path / "features.csv"
    destination = tmp_path / "scores.csv"
    _features().to_csv(source, index=False)
    process = subprocess.run(
        [sys.executable, "-m", "pepddg", "score-features", "--input", str(source), "--output", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert pd.read_csv(destination)["mutation"].tolist() == ["A1V", "A2L"]
    assert destination.with_suffix(".csv.json").exists()
