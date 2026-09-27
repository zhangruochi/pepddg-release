"""The release distribution exposes a standalone import and executable CLI."""

from __future__ import annotations

import subprocess
import sys


def test_public_import_exposes_scoring_api() -> None:
    from pepddg import PepDDGConfig, run_pepddg

    assert PepDDGConfig is not None
    assert callable(run_pepddg)


def test_public_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pepddg", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
