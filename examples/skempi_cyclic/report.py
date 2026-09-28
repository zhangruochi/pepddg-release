"""Summarize per-target SKEMPI cyclic smoke output and draw a comparison plot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


TARGETS = ("1SMF", "3EQS", "3EQY", "5XCO")


def rho(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        raise ValueError("Spearman correlation undefined for constant values")
    return float(spearmanr(a, b).statistic)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.references.parent
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest.get("schema") != "pepddg-skempi-cyclic-example/v1":
        raise ValueError("unknown cyclic example data manifest")
    for relative, expected in manifest.get("files_sha256", {}).items():
        path = data_root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"cyclic example data hash mismatch: {relative}")
    references = pd.read_csv(args.references)
    rows = []
    for target in TARGETS:
        reference = references.loc[references.target == target].copy()
        result = pd.read_csv(args.results / target / "scores.csv")
        merged = reference.merge(result[["target", "mutation", "rankscore_pepddg_zs"]],
                                 on=["target", "mutation"], validate="one_to_one")
        if len(merged) != len(reference) or len(merged) != len(result):
            raise ValueError(f"incomplete observation coverage for {target}")
        fresh_rho = rho(merged.ddg_exp, merged.rankscore_pepddg_zs)
        paper_rho = rho(merged.ddg_exp, merged.legacy_paper_score)
        order_agreement = rho(merged.legacy_paper_score, merged.rankscore_pepddg_zs)
        rows.append({"target": target, "n": len(merged), "fresh_experimental_rho": fresh_rho,
                     "paper_experimental_rho": paper_rho, "absolute_rho_drift": abs(fresh_rho-paper_rho),
                     "paper_score_agreement_rho": order_agreement})
    summary = pd.DataFrame(rows)
    paper = summary
    disulfide = summary[summary.target.isin(("1SMF", "5XCO"))]
    groups = {
        "paper_defined_four": {
            "targets": paper.target.tolist(),
            "mean_absolute_rho_drift": float(paper.absolute_rho_drift.mean()),
            "mean_paper_score_agreement": float(paper.paper_score_agreement_rho.mean()),
        },
        "true_disulfide_pair": {
            "targets": disulfide.target.tolist(),
            "mean_absolute_rho_drift": float(disulfide.absolute_rho_drift.mean()),
            "mean_paper_score_agreement": float(disulfide.paper_score_agreement_rho.mean()),
        },
    }
    for group in groups.values():
        group["pass"] = group["mean_absolute_rho_drift"] <= .10 and group["mean_paper_score_agreement"] >= .90
    target_pass = all(row["absolute_rho_drift"] <= .20 and row["paper_score_agreement_rho"] >= .70 for row in rows)
    report = {"schema": "pepddg-cyclic-smoke-report/v1", "data_manifest_sha256": hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest(), "targets": rows, "groups": groups,
              "metrics_pass": target_pass and all(group["pass"] for group in groups.values()),
              "interpretation": "Small-cohort engineering consistency check; not statistical equivalence or measured affinity."}
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "target_metrics.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = range(len(summary))
    ax.bar([i - .18 for i in x], summary.paper_experimental_rho, width=.36, label="Archived paper score")
    ax.bar([i + .18 for i in x], summary.fresh_experimental_rho, width=.36, label="Fresh PepDDG")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(list(x), summary.target)
    ax.set_ylabel("Spearman rho vs experimental ΔΔG")
    ax.set_title("SKEMPI v2.0 paper cyclic targets: per-target smoke")
    ax.legend(frameon=False)
    fig.savefig(args.output / "target_correlations.png", dpi=180)
    print(summary.to_string(index=False))
    return 0 if report["metrics_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
