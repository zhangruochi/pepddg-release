"""Summarize per-target SKEMPI cyclic smoke output and draw reproducible comparison and scatter plots."""
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



COLORS = {"1SMF": "#007F82", "5XCO": "#007F82", "3EQS": "#B76535", "3EQY": "#B76535"}
KINDS = {"1SMF": "Disulfide peptide", "5XCO": "Disulfide peptide · ACE/NH2 caps",
         "3EQS": "Linear PMI control", "3EQY": "Linear PMI control"}


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=240, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def draw_plots(cohorts: dict, summary: pd.DataFrame, output: Path) -> None:
    """Plot observations separately; score axes never imply calibrated energy."""
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#182D42", "axes.labelcolor": "#182D42",
                         "xtick.color": "#536579", "ytick.color": "#536579",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#D5DFE6", "svg.hashsalt": "pepddg-smoke-v1"}):
        for name, concordance in (("experiment_scatter", False), ("ranking_scatter", True)):
            fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.5))
            fig.subplots_adjust(left=.10, right=.97, bottom=.12, top=.84, hspace=.52, wspace=.28)
            title = "Do fresh scores track the experiments?" if not concordance else "How stable is the mutation ranking?"
            subtitle = "35 observations · four separate cohorts · seven paired restarts" if not concordance else "Archived paper scores vs fresh structure-to-score results"
            fig.text(.10, .965, title, fontsize=21, weight="bold", va="top")
            fig.text(.10, .917, subtitle, fontsize=11, color="#536579", va="top")
            for ax, target in zip(axes.flat, TARGETS):
                frame = cohorts[target]
                metric = summary.loc[summary.target == target].iloc[0]
                x = frame.legacy_paper_score.rank(method="average", pct=True) if concordance else frame.ddg_exp
                y = frame.rankscore_pepddg_zs.rank(method="average", pct=True) if concordance else frame.rankscore_pepddg_zs
                color = COLORS[target]
                if concordance:
                    ax.plot([0, 1.04], [0, 1.04], color="#C5D1DA", lw=1.2, ls="--", zorder=1)
                    ax.set_xlim(0, 1.04)
                ax.scatter(x, y, s=72, c=color, edgecolors="white", linewidths=1.2, zorder=3)
                ax.set_ylim(-.04, 1.08)
                ax.set_yticks([0, .25, .5, .75, 1])
                ax.margins(x=.15)
                ax.grid(axis="y", color="#E9EEF2", linewidth=.8, zorder=0)
                ax.set_axisbelow(True)
                ax.set_title(f"{target}  /  {KINDS[target]}", loc="left", fontsize=11, pad=12, weight="bold", color=color)
                correlation = metric.paper_score_agreement_rho if concordance else metric.fresh_experimental_rho
                ax.text(.03, .94, f"n = {len(frame)}   Spearman ρ = {correlation:.3f}", transform=ax.transAxes, va="top", fontsize=10,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
                ax.set_xlabel("Archived score percentile" if concordance else "Experimental ΔΔG (kcal/mol)")
                ax.set_ylabel("Fresh score percentile" if concordance else "Fresh PepDDG rank score")
            foot = "Percentiles are computed within each target. Dashed line: identical ranks." if concordance else "Lower scores favor a mutation within its cohort; the score is not an energy or affinity."
            fig.text(.10, .04, foot, fontsize=10, color="#536579")
            save_figure(fig, output, name)
        fig, ax = plt.subplots(figsize=(10.8, 4.8))
        fig.subplots_adjust(left=.30, right=.93, top=.74, bottom=.18)
        fig.text(.07, .94, "Paper-to-release consistency", fontsize=21, weight="bold", va="top")
        fig.text(.07, .85, "Spearman correlation with experimental ΔΔG · separate target cohorts", color="#536579", fontsize=11)
        for i, row in summary.iterrows():
            ax.plot([row.paper_experimental_rho, row.fresh_experimental_rho], [i, i], color="#CED8E0", lw=4, zorder=1)
            ax.scatter(row.paper_experimental_rho, i, s=80, color="#8293A4", edgecolor="white", zorder=3,
                       label="Archived paper score" if i == 0 else None)
            ax.scatter(row.fresh_experimental_rho, i, s=90, color="#007F82", edgecolor="white", zorder=4,
                       label="Fresh PepDDG" if i == 0 else None)
            ax.annotate(f"{row.paper_experimental_rho:.3f}", (row.paper_experimental_rho, i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color="#637588")
            ax.annotate(f"{row.fresh_experimental_rho:.3f}", (row.fresh_experimental_rho, i), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=9, color="#007F82", weight="bold")
        ax.set_yticks(range(len(summary)), [f"{r.target}   ·   n = {r.n}" for r in summary.itertuples()])
        ax.invert_yaxis()
        ax.set_ylim(3.6, -.6)
        ax.set_xlim(0, 1.03)
        ax.set_xlabel("Spearman ρ")
        ax.grid(axis="x", color="#E9EEF2")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.06, .79), ncol=2, frameon=False)
        save_figure(fig, output, "target_correlations")

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
    cohorts = {}
    for target in TARGETS:
        reference = references.loc[references.target == target].copy()
        result = pd.read_csv(args.results / target / "scores.csv")
        merged = reference.merge(result[["target", "mutation", "rankscore_pepddg_zs"]],
                                 on=["target", "mutation"], validate="one_to_one")
        if len(merged) != len(reference) or len(merged) != len(result):
            raise ValueError(f"incomplete observation coverage for {target}")
        cohorts[target] = merged
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
    draw_plots(cohorts, summary, args.output)
    print(summary.to_string(index=False))
    return 0 if report["metrics_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
