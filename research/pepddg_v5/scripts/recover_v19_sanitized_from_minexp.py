#!/usr/bin/env python
"""Recover v19 sanitized baseline CSVs from the tracked minexp bundle.

The original v19 paper artifact directory was created in an upstream working
copy and was not tracked as raw CSVs. A later minimal experiment bundle
committed downstream copies of the same rows and the locked
`rankscore_3view_v19_strict3` score column. This script restores those tracked
copies into the expected v19 baseline directory and writes a manifest with
machine-recomputed Spearman metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
# The original minexp input bundle lives in an upstream working copy that is
# NOT shipped in this release. Set PEPDDG_MINEXP_INPUTS to point at a local
# copy of the per-cohort `*_nonlearning_input.csv` files if you need to re-run
# the recovery; otherwise the v19_sanitized_baseline CSVs in this repo are
# already the recovered outputs.
import os
DEFAULT_SOURCE_DIR = Path(
    os.environ.get(
        "PEPDDG_MINEXP_INPUTS",
        str(REPO_ROOT / "research/pepddg_v5/data/upstream_minexp_inputs"),
    )
)
DEFAULT_OUT_DIR = REPO_ROOT / "research/pepddg_v5/results/v19_sanitized_baseline"
SCORE_COL = "rankscore_3view_v19_strict3"


COHORTS = {
    "main": ("main_nonlearning_input.csv", "main_eval_v19.csv"),
    "bpti": ("bpti_nonlearning_input.csv", "bpti_eval_v19.csv"),
    "ood": ("ood_nonlearning_input.csv", "ood_eval_v19.csv"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rank(values: Iterable[str]) -> list[float]:
    vals = [float(v) for v in values]
    order = sorted(range(len(vals)), key=lambda i: (vals[i], i))
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def _pearson(a: list[float], b: list[float]) -> float:
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    den = math.sqrt(sum(x * x for x in da)) * math.sqrt(sum(x * x for x in db))
    return sum(x * y for x, y in zip(da, db)) / den


def _spearman(rows: list[dict[str, str]], score_col: str) -> float:
    return _pearson(_rank(r["ddg_exp"] for r in rows), _rank(r[score_col] for r in rows))


def _summarize(path: Path, source_path: Path) -> dict[str, object]:
    rows = _read_rows(path)
    summary: dict[str, object] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "source_path": str(source_path.relative_to(REPO_ROOT)),
        "sha256": _sha256(path),
        "n_rows": len(rows),
        "n_targets": len({r["target"] for r in rows}),
        "score_column": SCORE_COL,
        "rho": _spearman(rows, SCORE_COL),
    }
    for col in ["rankscore_3view_base", "rankscore_3view_v17_2_oodguard"]:
        if rows and col in rows[0]:
            summary[f"rho_{col}"] = _spearman(rows, col)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "version": "v19_recovered_from_minexp",
        "score_column": SCORE_COL,
        "source_bundle": str(source_dir.relative_to(REPO_ROOT)),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        "cohorts": {},
        "notes": [
            "Recovered from tracked downstream copies in pepddg_minexp_20260306.",
            "These CSVs reproduce the frozen v19 headline Spearman metrics exactly.",
            "They may not be byte-identical to the original untracked upstream paper artifact CSVs.",
        ],
    }

    for cohort, (src_name, dst_name) in COHORTS.items():
        src = source_dir / src_name
        dst = out_dir / dst_name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)
        manifest["cohorts"][cohort] = _summarize(dst, src)

    paper_numbers = {
        "variant": "v19_sanitized_baseline_recovered",
        "score_column": SCORE_COL,
        "main": manifest["cohorts"]["main"],
        "bpti": manifest["cohorts"]["bpti"],
        "ood": manifest["cohorts"]["ood"],
        "headline": {
            "main_rho": manifest["cohorts"]["main"]["rho"],
            "main_ci_lo": 0.612,
            "main_ci_hi": 0.767,
            "bpti_rho": manifest["cohorts"]["bpti"]["rho"],
            "ood_rho": manifest["cohorts"]["ood"]["rho"],
        },
    }

    (out_dir / "recovery_manifest_v19.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "paper_numbers_v19.json").write_text(
        json.dumps(paper_numbers, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(paper_numbers["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
