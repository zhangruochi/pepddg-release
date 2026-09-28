"""Replay the released 332-row rank regression with labels evaluator-only.

This is B1a frozen-feature evidence, never a structural reproduction.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import PepDDGConfig
from .release_numbers import (
    CANONICAL_JSON, DATASETS, ZS_SCORE_COLUMN, ZS_SOURCE_COLUMN,
    _row_identity_hash, _target_bootstrap_ci, sha256_file,
)
from .scoring import score_dataframe


def replay_frozen_skempi(repo_root: str | Path, output_dir: str | Path) -> dict:
    """Check the frozen ZS score vector, pooled rho and target-bootstrap CI."""
    root = Path(repo_root).resolve()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("frozen replay output directory must be absent or empty")
    source_path = root / DATASETS["main"]
    reference_path = root / CANONICAL_JSON
    reference = json.loads(reference_path.read_text())
    source = pd.read_csv(source_path)
    cohort = reference["cohort_identity"]["datasets"]["main"]
    expected = reference["modes"]["zs"]["main"]
    if sha256_file(source_path) != cohort["sha256"] or _row_identity_hash(source) != cohort["row_identity_hash"]:
        raise ValueError("frozen source bytes or row identity differ from the release record")
    if len(source) != 332 or source["target"].nunique() != 33:
        raise ValueError("frozen 332-row / 33-target membership is incomplete")
    required = ["target", ZS_SOURCE_COLUMN, "ddg_exp"]
    if source[required].isna().any().any():
        raise ValueError("frozen source has missing predictor anchor or evaluator label")

    # The predictor receives no experimental label or recorded output score.
    predictor_input = source[["target", ZS_SOURCE_COLUMN]].copy()
    scored, details = score_dataframe(
        predictor_input,
        PepDDGConfig(input_csv=str(source_path), output_csv=str(output / "scores.csv"),
                     mode="zs", score_column=ZS_SCORE_COLUMN),
    )
    if details["anchor_source"] != ZS_SOURCE_COLUMN:
        raise ValueError("frozen replay used the wrong score anchor")
    anchor = source[ZS_SOURCE_COLUMN].to_numpy(dtype=float)
    order = np.argsort(anchor, kind="mergesort")
    independently_ranked = np.empty(len(source), dtype=float)
    independently_ranked[order] = np.arange(len(source), dtype=float) / float(len(source) - 1)
    score = scored[ZS_SCORE_COLUMN].to_numpy(dtype=float)
    max_delta = float(np.max(np.abs(score - independently_ranked)))

    evaluator = pd.DataFrame({
        "target": source["target"], "ddg_exp": source["ddg_exp"],
        ZS_SCORE_COLUMN: score,
    })
    rho = float(spearmanr(evaluator[ZS_SCORE_COLUMN], evaluator["ddg_exp"]).statistic)
    ci_lo, ci_hi = _target_bootstrap_ci(
        evaluator, ZS_SCORE_COLUMN, label=f"main:zs:{ZS_SCORE_COLUMN}"
    )
    rho_delta = abs(rho - float(expected["rho"]))
    ci_max_delta = max(abs(ci_lo - float(expected["ci_lo"])), abs(ci_hi - float(expected["ci_hi"])))
    passed = max_delta <= 1e-12 and rho_delta <= 1e-12 and ci_max_delta <= 1e-10

    output.mkdir(parents=True, exist_ok=True)
    score_table = scored[["target", ZS_SOURCE_COLUMN, ZS_SCORE_COLUMN]].copy()
    score_table.insert(1, "original_row_index", np.arange(len(score_table)))
    score_table.to_csv(output / "scores.csv", index=False)
    report = {
        "status": "PASS_B1A_FROZEN_ONLY" if passed else "FAIL_B1A_FROZEN_ONLY",
        "scope": "published_anchor_rank_regression_not_structure_to_score",
        "n_rows": len(source), "n_targets": int(source["target"].nunique()),
        "predictor_input_columns": list(predictor_input.columns),
        "source_sha256": sha256_file(source_path),
        "reference_sha256": sha256_file(reference_path),
        "scores_sha256": sha256_file(output / "scores.csv"),
        "scoring_sha256": sha256((Path(__file__).parent / "scoring.py").read_bytes()).hexdigest(),
        "max_score_vector_delta": max_delta,
        "rho": rho, "rho_reference": float(expected["rho"]), "rho_delta": rho_delta,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "ci_reference": [float(expected["ci_lo"]), float(expected["ci_hi"])],
        "ci_max_delta": ci_max_delta,
        "tolerances": {"score_vector": 1e-12, "rho": 1e-12, "ci": 1e-10},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay_frozen_skempi(args.repo_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS_B1A_FROZEN_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
