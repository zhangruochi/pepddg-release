"""Frozen SKEMPI replay is a regression, with experimental labels evaluator-only."""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_replay_matches_recorded_332_row_numbers(tmp_path: Path) -> None:
    from pepddg.frozen_benchmark import replay_frozen_skempi

    report = replay_frozen_skempi(REPO_ROOT, tmp_path / "result")
    assert report["status"] == "PASS_B1A_FROZEN_ONLY"
    assert report["n_rows"] == 332
    assert report["n_targets"] == 33
    assert report["predictor_input_columns"] == ["target", "rankscore_3view_base"]
    assert report["max_score_vector_delta"] <= 1e-12
    assert report["rho_delta"] <= 1e-12
    assert report["ci_max_delta"] <= 1e-10
    assert (tmp_path / "result" / "scores.csv").is_file()


def test_frozen_replay_rejects_existing_output(tmp_path: Path) -> None:
    from pepddg.frozen_benchmark import replay_frozen_skempi

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "other.txt").write_text("preserve me")
    with pytest.raises(FileExistsError):
        replay_frozen_skempi(REPO_ROOT, output)
