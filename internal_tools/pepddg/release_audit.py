"""Audit PepDDG sealed release artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .release_numbers import (
    CAL_SCORE_COLUMN,
    CANONICAL_JSON,
    LEGACY_MANIFEST,
    RELEASE_ID,
    RELEASE_MANIFEST,
    ZS_SCORE_COLUMN,
    build_release_numbers,
    sha256_file,
)


V19_EXPECTED = {
    "score_column": "rankscore_3view_v19_strict3",
    "headline": {
        "main_rho": 0.6908636100957133,
        "main_ci_lo": 0.612,
        "main_ci_hi": 0.767,
        "bpti_rho": 0.7121243607250747,
        "ood_rho": 0.23534660137316027,
    },
    "cohorts": {
        "main": {"n_rows": 332, "n_targets": 33},
        "bpti": {"n_rows": 456, "n_targets": 2},
        "ood": {"n_rows": 745, "n_targets": 4},
    },
}


def _assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def audit_release(repo_root: Path) -> dict[str, Any]:
    """Audit active ZS+Cal release numbers + headline expectations."""
    metrics_path = repo_root / CANONICAL_JSON
    release_manifest_path = repo_root / RELEASE_MANIFEST
    legacy_manifest_path = repo_root / LEGACY_MANIFEST
    if not metrics_path.exists():
        raise AssertionError(f"canonical metrics file is missing: {CANONICAL_JSON}")
    if not release_manifest_path.exists():
        raise AssertionError(f"release manifest is missing: {RELEASE_MANIFEST}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = build_release_numbers(repo_root, created_utc=metrics["created_utc"])
    if metrics != expected:
        raise AssertionError("paper_numbers_zs_cal.json does not match recomputation from source CSVs")

    _assert_equal(metrics["release_id"], RELEASE_ID, "release_id")
    _assert_equal(metrics["headline"]["main_mode"], "zs", "headline.main_mode")
    _assert_equal(metrics["headline"]["main_score_column"], ZS_SCORE_COLUMN, "headline.main_score_column")
    _assert_equal(metrics["headline"]["cal_score_column"], CAL_SCORE_COLUMN, "headline.cal_score_column")

    for mode in ["zs", "cal"]:
        for dataset in ["main", "bpti", "ood"]:
            block = metrics["modes"][mode][dataset]
            csv_path = repo_root / block["source_path"]
            _assert_equal(sha256_file(csv_path), block["sha256"], f"{mode}.{dataset}.sha256")

    manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    _assert_equal(manifest["release_id"], RELEASE_ID, "manifest.release_id")
    _assert_equal(manifest["canonical_metrics_source"], CANONICAL_JSON, "manifest.canonical_metrics_source")
    _assert_equal(
        manifest["canonical_metrics_sha256"],
        sha256_file(metrics_path),
        "manifest.canonical_metrics_sha256",
    )

    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    active = legacy_manifest.get("active_release", {})
    _assert_equal(active.get("canonical_metrics_source"), CANONICAL_JSON, "legacy.active_release.canonical_metrics_source")
    _assert_equal(active.get("release_manifest"), RELEASE_MANIFEST, "legacy.active_release.release_manifest")
    if "paper_numbers_v19.json" in legacy_manifest.get("official_policy", ""):
        if "historical" not in legacy_manifest["official_policy"].lower():
            raise AssertionError("legacy manifest still treats paper_numbers_v19.json as active policy")

    # Independent v19 historical-headline cross-check using paper_numbers_v19.json
    # (this file is shipped as a frozen Cal-only baseline; we sanity-check that
    # its row counts and headline rho match the V19_EXPECTED constants and that
    # its source CSVs hash to the recorded sha256).
    legacy_metrics_path = repo_root / "research/pepddg_v5/results/v19_sanitized_baseline/paper_numbers_v19.json"
    legacy_metrics = json.loads(legacy_metrics_path.read_text(encoding="utf-8"))
    _assert_equal(legacy_metrics["score_column"], V19_EXPECTED["score_column"], "v19.score_column")
    for key, expected_val in V19_EXPECTED["headline"].items():
        _assert_close(float(legacy_metrics["headline"][key]), expected_val, f"v19.headline.{key}")
    legacy_checked_files: list[str] = []
    for cohort, expected_counts in V19_EXPECTED["cohorts"].items():
        observed = legacy_metrics[cohort]
        for key, expected_val in expected_counts.items():
            _assert_equal(int(observed[key]), expected_val, f"v19.{cohort}.{key}")
        csv_path = repo_root / observed["path"]
        digest = sha256_file(csv_path)
        _assert_equal(digest, observed["sha256"], f"v19.{cohort}.sha256")
        legacy_checked_files.append(str(csv_path.relative_to(repo_root)))

    return {
        "status": "ok",
        "release_id": RELEASE_ID,
        "canonical_metrics_source": CANONICAL_JSON,
        "release_manifest": RELEASE_MANIFEST,
        "legacy_manifest": LEGACY_MANIFEST,
        "headline": metrics["headline"],
        "checked_modes": ["zs", "cal"],
        "checked_datasets": ["main", "bpti", "ood"],
        "legacy_v19_checked_files": legacy_checked_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit active PepDDG ZS+Cal release artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    result = audit_release(Path(args.repo_root).resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"PepDDG release audit: {result['status']}")
        print(f"metrics: {result['canonical_metrics_source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
