"""Read-only dependency and bundled-resource diagnostics."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata, util
from pathlib import Path
import sys


_DEPENDENCIES = {
    "numpy": ("numpy", "numpy"),
    "pandas": ("pandas", "pandas"),
    "scipy": ("scipy", "scipy"),
    "pyyaml": ("yaml", "PyYAML"),
    "gemmi": ("gemmi", "gemmi"),
    "openmm": ("openmm", "OpenMM"),
    "pdbfixer": ("pdbfixer", "pdbfixer"),
    "torch": ("torch", "torch"),
}


def diagnose_runtime() -> dict:
    """Report package presence without importing models or running predictions."""
    dependencies: dict[str, dict] = {}
    for name, (module, distribution) in _DEPENDENCIES.items():
        available = util.find_spec(module) is not None
        try:
            version = metadata.version(distribution) if available else None
        except metadata.PackageNotFoundError:
            version = None
        dependencies[name] = {"available": available, "version": version}

    checkpoint = Path(__file__).parent / "_backend/model_weights/v_48_020.pt"
    model = {
        "available": checkpoint.is_file(),
        "sha256": sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None,
    }
    structural_names = ("gemmi", "openmm", "pdbfixer", "torch")
    return {
        "schema": "pepddg-doctor/v1",
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "feature_scoring_ready": all(dependencies[name]["available"] for name in ("numpy", "pandas", "scipy", "pyyaml")),
        "structural_runtime_ready": model["available"] and all(dependencies[name]["available"] for name in structural_names),
        "dependencies": dependencies,
        "model_checkpoint": model,
        "scope": "presence_only_no_runtime_execution",
    }
