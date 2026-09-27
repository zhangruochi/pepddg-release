"""The release distribution exposes a standalone import and executable CLI."""

from __future__ import annotations

import subprocess
import sys
import json


def test_public_import_exposes_scoring_api() -> None:
    from pepddg import PepDDGConfig, diagnose_runtime, run_pepddg

    assert PepDDGConfig is not None
    assert callable(run_pepddg)
    assert diagnose_runtime()["schema"] == "pepddg-doctor/v1"


def test_public_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pepddg", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_doctor_reports_runtime_without_running_prediction() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pepddg", "doctor", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "pepddg-doctor/v1"
    assert report["feature_scoring_ready"] is True
    assert isinstance(report["structural_runtime_ready"], bool)
    assert report["dependencies"]["numpy"]["available"] is True
    assert report["dependencies"]["openmm"]["available"] in {True, False}
    assert report["model_checkpoint"]["available"] is True
    assert len(report["model_checkpoint"]["sha256"]) == 64
