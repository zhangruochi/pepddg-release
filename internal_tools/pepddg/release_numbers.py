"""Build canonical PepDDG ZS+Cal release-number artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PepDDGConfig
from .scoring import score_dataframe


SCHEMA_VERSION = "pepddg-zs-cal-release-v1"
RELEASE_ID = "2026-04-23-zs-cal-neurips"
BOOTSTRAP_SEED = 20260302
BOOTSTRAP_RESAMPLES = 1000
DATASETS = {
    "main": "research/pepddg_v5/results/v19_sanitized_baseline/main_eval_v19.csv",
    "bpti": "research/pepddg_v5/results/v19_sanitized_baseline/bpti_eval_v19.csv",
    "ood": "research/pepddg_v5/results/v19_sanitized_baseline/ood_eval_v19.csv",
}
RELEASE_DIR = "research/pepddg_v5/results/zs_cal_neurips_release"
CANONICAL_JSON = f"{RELEASE_DIR}/paper_numbers_zs_cal.json"
RELEASE_MANIFEST = f"{RELEASE_DIR}/release_manifest_zs_cal.json"
LEGACY_MANIFEST = "research/pepddg_v5/results/v19_release_manifest.json"
ZS_SCORE_COLUMN = "rankscore_pepddg_zs"
ZS_SOURCE_COLUMN = "rankscore_3view_base"
CAL_SCORE_COLUMN = "rankscore_3view_v19_strict3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spearman(score: pd.Series, truth: pd.Series) -> float:
    valid = score.notna() & truth.notna()
    if int(valid.sum()) < 2:
        raise ValueError("Spearman requires at least two valid rows.")
    return float(spearmanr(score[valid], truth[valid]).statistic)


def _bootstrap_seed(label: str) -> int:
    offset = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return int((BOOTSTRAP_SEED + offset) % (2**32 - 1))


def _target_bootstrap_ci(df: pd.DataFrame, score_column: str, *, label: str) -> tuple[float, float]:
    valid = df[[score_column, "ddg_exp", "target"]].dropna()
    targets = np.array(sorted(valid["target"].astype(str).unique()))
    if len(targets) < 1 or len(valid) < 2:
        raise ValueError(f"Cannot bootstrap {label}: insufficient valid rows.")

    by_target = {target: valid[valid["target"].astype(str) == target] for target in targets}
    rng = np.random.default_rng(_bootstrap_seed(label))
    values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        sample = pd.concat([by_target[target] for target in sampled_targets], ignore_index=True)
        values.append(_spearman(sample[score_column], sample["ddg_exp"]))
    lo, hi = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _calibration_counts(df: pd.DataFrame) -> dict[str, int] | None:
    if "is_calibration" not in df.columns:
        return None
    counts = df["is_calibration"].astype(bool).value_counts(dropna=False).to_dict()
    return {
        "true": int(counts.get(True, 0)),
        "false": int(counts.get(False, 0)),
    }


def _target_list_hash(df: pd.DataFrame) -> str:
    targets = sorted(str(x) for x in df["target"].dropna().unique())
    return sha256_text(json.dumps(targets, separators=(",", ":"), ensure_ascii=True))


def _row_identity_hash(df: pd.DataFrame) -> str:
    payload = [
        {"target": str(target), "original_row_index": int(i)}
        for i, target in enumerate(df["target"].tolist())
    ]
    return sha256_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))


def _prepare_dataset(repo_root: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rel_path = DATASETS[dataset]
    path = repo_root / rel_path
    df = pd.read_csv(path)
    if "target" not in df.columns or "ddg_exp" not in df.columns:
        raise ValueError(f"{rel_path} must contain target and ddg_exp columns.")
    if ZS_SOURCE_COLUMN not in df.columns:
        raise ValueError(f"{rel_path} is missing {ZS_SOURCE_COLUMN}.")
    if CAL_SCORE_COLUMN not in df.columns:
        raise ValueError(f"{rel_path} is missing {CAL_SCORE_COLUMN}.")

    df = df.copy()
    scored_df, scoring_details = score_dataframe(
        df,
        PepDDGConfig(
            input_csv=rel_path,
            output_csv=f"{dataset}_zs.csv",
            mode="zs",
            score_column=ZS_SCORE_COLUMN,
        ),
    )
    df[ZS_SCORE_COLUMN] = scored_df[ZS_SCORE_COLUMN]
    valid_rows = int((df[["ddg_exp", ZS_SCORE_COLUMN, CAL_SCORE_COLUMN]].notna().all(axis=1)).sum())
    metadata: dict[str, Any] = {
        "source_path": rel_path,
        "sha256": sha256_file(path),
        "n_rows": int(len(df)),
        "valid_rows_for_all_modes": valid_rows,
        "n_targets": int(df["target"].nunique(dropna=True)),
        "target_list_hash": _target_list_hash(df),
        "row_identity_schema": ["target", "original_row_index"],
        "row_identity_hash": _row_identity_hash(df),
        "zs_production_scoring": {
            "mode": scoring_details["mode"],
            "score_column": scoring_details["score_column"],
            "anchor_source": scoring_details["anchor_source"],
            "used_columns": scoring_details["used_columns"],
        },
    }
    return df, metadata


def _mode_block(df: pd.DataFrame, metadata: dict[str, Any], *, dataset: str, mode: str) -> dict[str, Any]:
    score_column = ZS_SCORE_COLUMN if mode == "zs" else CAL_SCORE_COLUMN
    rho = _spearman(df[score_column], df["ddg_exp"])
    ci_lo, ci_hi = _target_bootstrap_ci(df, score_column, label=f"{dataset}:{mode}:{score_column}")
    block: dict[str, Any] = {
        "score_column": score_column,
        "rho": rho,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_rows": metadata["n_rows"],
        "n_valid_rows": int(df[[score_column, "ddg_exp"]].dropna().shape[0]),
        "n_targets": metadata["n_targets"],
        "source_path": metadata["source_path"],
        "sha256": metadata["sha256"],
    }
    counts = _calibration_counts(df)
    if counts is not None:
        block["is_calibration_counts"] = counts
    if mode == "zs":
        block["source_score_column"] = ZS_SOURCE_COLUMN
        block["derivation"] = f"{ZS_SCORE_COLUMN}=stable_rank({ZS_SOURCE_COLUMN})"
    else:
        block["variant"] = "pepddg_cal_v19"
    return block


def build_release_numbers(repo_root: Path, *, created_utc: str | None = None) -> dict[str, Any]:
    """Build the in-memory canonical release-number object."""
    created_utc = created_utc or _utc_now()
    datasets: dict[str, pd.DataFrame] = {}
    cohort_identity: dict[str, Any] = {
        "row_key_schema": ["target", "original_row_index"],
        "row_key_note": "Source CSVs do not expose mutation identifiers; row identity is audited by target plus original row index.",
        "datasets": {},
    }
    for dataset in DATASETS:
        df, metadata = _prepare_dataset(repo_root, dataset)
        datasets[dataset] = df
        cohort_identity["datasets"][dataset] = metadata

    modes = {
        "zs": {dataset: _mode_block(datasets[dataset], cohort_identity["datasets"][dataset], dataset=dataset, mode="zs") for dataset in DATASETS},
        "cal": {dataset: _mode_block(datasets[dataset], cohort_identity["datasets"][dataset], dataset=dataset, mode="cal") for dataset in DATASETS},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "created_utc": created_utc,
        "metric_method": {
            "correlation": "scipy.stats.spearmanr",
            "score_rank_policy": "stable normalized ascending ranks for ZS derivation",
            "spearman_tie_policy": "scipy average ranks",
            "confidence_interval": "target-stratified bootstrap percentile 95%",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed_derivation": "bootstrap_seed + sha256(dataset:mode:score_column) prefix",
            "bootstrap_resample_unit": "target",
        },
        "cohort_identity": cohort_identity,
        "modes": modes,
        "headline": {
            "main_mode": "zs",
            "main_score_column": ZS_SCORE_COLUMN,
            "main_rho": modes["zs"]["main"]["rho"],
            "main_ci_lo": modes["zs"]["main"]["ci_lo"],
            "main_ci_hi": modes["zs"]["main"]["ci_hi"],
            "cal_score_column": CAL_SCORE_COLUMN,
            "cal_main_rho": modes["cal"]["main"]["rho"],
            "cal_main_ci_lo": modes["cal"]["main"]["ci_lo"],
            "cal_main_ci_hi": modes["cal"]["main"]["ci_hi"],
            "zs_bpti_rho": modes["zs"]["bpti"]["rho"],
            "zs_ood_rho": modes["zs"]["ood"]["rho"],
            "cal_bpti_rho": modes["cal"]["bpti"]["rho"],
            "cal_ood_rho": modes["cal"]["ood"]["rho"],
        },
    }


def build_release_manifest(repo_root: Path, numbers: dict[str, Any]) -> dict[str, Any]:
    canonical_path = repo_root / CANONICAL_JSON
    legacy_manifest_path = repo_root / LEGACY_MANIFEST
    legacy_manifest_sha = sha256_file(legacy_manifest_path) if legacy_manifest_path.exists() else None
    source_files = [metadata["source_path"] for metadata in numbers["cohort_identity"]["datasets"].values()]
    return {
        "release_id": RELEASE_ID,
        "created_utc": numbers["created_utc"],
        "active_algorithm": "PepDDG-ZS",
        "calibrated_ablation": "PepDDG-Cal",
        "canonical_metrics_source": CANONICAL_JSON,
        "canonical_metrics_sha256": sha256_file(canonical_path) if canonical_path.exists() else None,
        "source_csvs": source_files,
        "legacy_v19_manifest": LEGACY_MANIFEST,
        "legacy_v19_manifest_sha256_before_update": legacy_manifest_sha,
        "legacy_v19_metrics_source": "research/pepddg_v5/results/v19_sanitized_baseline/paper_numbers_v19.json",
        "official_policy": "NeurIPS paper numbers must come from paper_numbers_zs_cal.json; paper_numbers_v19.json is historical Cal-only evidence.",
    }


def write_release_artifacts(repo_root: Path, *, created_utc: str | None = None) -> dict[str, Any]:
    numbers = build_release_numbers(repo_root, created_utc=created_utc)
    release_dir = repo_root / RELEASE_DIR
    release_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = repo_root / CANONICAL_JSON
    canonical_path.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_release_manifest(repo_root, numbers)
    manifest["canonical_metrics_sha256"] = sha256_file(canonical_path)
    manifest_path = repo_root / RELEASE_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    legacy_manifest_path = repo_root / LEGACY_MANIFEST
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["active_release"] = {
        "release_id": RELEASE_ID,
        "canonical_metrics_source": CANONICAL_JSON,
        "release_manifest": RELEASE_MANIFEST,
        "official_conclusion_dir": RELEASE_DIR,
    }
    legacy_manifest["historical_v19"] = {
        "canonical_metrics_source": "research/pepddg_v5/results/v19_sanitized_baseline/paper_numbers_v19.json",
        "official_conclusion_dir": "research/pepddg_v5/results/v19_sanitized_baseline",
        "status": "historical_calibrated_baseline",
    }
    legacy_manifest["official_policy"] = (
        "Active NeurIPS paper numbers must reference "
        "research/pepddg_v5/results/zs_cal_neurips_release/paper_numbers_zs_cal.json; "
        "v19_sanitized_baseline/paper_numbers_v19.json is historical Cal-only evidence."
    )
    legacy_manifest.setdefault("notes", [])
    if not any("PepDDG-ZS is the active zero-shot manuscript algorithm" in n for n in legacy_manifest["notes"]):
        legacy_manifest["notes"].append(
            "PepDDG-ZS is the active zero-shot manuscript algorithm; PepDDG-Cal/v19 is retained as a calibrated ablation."
        )
    legacy_manifest_path.write_text(json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "numbers": numbers,
        "manifest": manifest,
        "legacy_manifest": legacy_manifest,
        "canonical_path": str(canonical_path.relative_to(repo_root)),
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "legacy_manifest_path": str(legacy_manifest_path.relative_to(repo_root)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical PepDDG ZS+Cal release numbers.")
    parser.add_argument("--repo-root", default=".", help="Repository root path.")
    parser.add_argument("--json", action="store_true", help="Print generated artifact summary as JSON.")
    args = parser.parse_args(argv)

    result = write_release_artifacts(Path(args.repo_root).resolve())
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k not in {"numbers"}}, indent=2, sort_keys=True))
    else:
        print(f"Wrote {result['canonical_path']}")
        print(f"Wrote {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
