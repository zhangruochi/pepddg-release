#!/usr/bin/env python
"""Assemble and evaluate PepDDG v16 (Physical 2.0 + Structure 2.0)."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pepddg_v5.scripts.v12_followup_common import pareto_frontier_3d  # noqa: E402
from research.pepddg_v5.scripts.v14_ab_common import gate_summary  # noqa: E402
from research.pepddg_v5.scripts.v16_ps2s2_common import (  # noqa: E402
    HIST_PHYS_COLS,
    V16Config,
    add_ps2s2_channels,
    apply_v16_variant,
    cfg_to_dict,
    eval_prediction,
    get_v16_used_columns,
    strict3_policy_audit,
)


def _triplet_summary(
    main_df: pd.DataFrame,
    bpti_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    pred_col: str,
    gate_main_rho: float,
    gate_external_delta: float,
) -> dict:
    m_main = eval_prediction(main_df, pred_col)
    m_bpti = eval_prediction(bpti_df, pred_col)
    m_ood = eval_prediction(ood_df, pred_col)
    return gate_summary(
        m_main,
        m_bpti,
        m_ood,
        gate_main_rho=gate_main_rho,
        gate_external_delta=gate_external_delta,
    )


def _merge_main_hist_physics(df: pd.DataFrame, hist_csv: str) -> pd.DataFrame:
    out = df.copy()
    hist_path = Path(hist_csv)
    if not hist_path.exists():
        for c in HIST_PHYS_COLS:
            if c not in out.columns:
                out[c] = pd.NA
        return out
    hist = pd.read_csv(hist_path)
    num_cols = [c for c in hist.columns if pd.api.types.is_numeric_dtype(hist[c])]
    hist = hist.groupby(["target", "mut"], as_index=False)[num_cols].mean()
    keep = [c for c in HIST_PHYS_COLS if c in hist.columns]
    if not keep:
        for c in HIST_PHYS_COLS:
            if c not in out.columns:
                out[c] = pd.NA
        return out
    out = out.merge(hist[["target", "mut"] + keep], on=["target", "mut"], how="left", suffixes=("", "_hist"))
    for c in keep:
        hc = f"{c}_hist"
        if hc in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(pd.to_numeric(out[hc], errors="coerce"))
            out = out.drop(columns=[hc])
    for c in HIST_PHYS_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def _ensure_hist_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in HIST_PHYS_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-csv", default="research/pepddg_v5/results/v13_elegant/main_eval_v13.csv")
    parser.add_argument("--bpti-csv", default="research/pepddg_v5/results/v13_elegant/bpti_eval_v13.csv")
    parser.add_argument("--ood-csv", default="research/pepddg_v5/results/v13_elegant/ood_eval_v13.csv")
    parser.add_argument("--hist-physics-csv", default="research/pepddg_v5/results/all_scores_predicted.csv")
    parser.add_argument("--out-dir", default="research/pepddg_v5/results/v16_ps2s2")
    parser.add_argument("--core-col", default="rankscore_3view_v13_v13_cons")
    parser.add_argument("--fallback-col", default="rankscore_3view_base")
    parser.add_argument(
        "--enable-main-only-hist",
        action="store_true",
        help="Enable asymmetric main-only historical physics enrichment (exploratory only).",
    )
    parser.add_argument("--gate-main-rho", type=float, default=0.680)
    parser.add_argument("--gate-external-delta", type=float, default=-0.010)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(args.main_csv)
    bpti_df = pd.read_csv(args.bpti_csv)
    ood_df = pd.read_csv(args.ood_csv)

    if args.enable_main_only_hist:
        main_df = _merge_main_hist_physics(main_df, args.hist_physics_csv)
        bpti_df = _ensure_hist_cols(bpti_df)
        ood_df = _ensure_hist_cols(ood_df)
    else:
        for df in [main_df, bpti_df, ood_df]:
            for c in HIST_PHYS_COLS:
                df[c] = pd.NA

    # Strict clean-3 policy.
    all_input_cols = sorted(set(main_df.columns) | set(bpti_df.columns) | set(ood_df.columns))
    used_cols = sorted(set(get_v16_used_columns() + [args.core_col, args.fallback_col]))
    policy = strict3_policy_audit(
        used_columns=used_cols,
        input_columns=all_input_cols,
    )
    (out_dir / "policy_audit.json").write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    if not policy["pass"]:
        raise ValueError(f"Strict3 policy failed: {policy['used_columns_violations']}")

    main_df = add_ps2s2_channels(main_df)
    bpti_df = add_ps2s2_channels(bpti_df)
    ood_df = add_ps2s2_channels(ood_df)
    for name, df in [("main", main_df), ("bpti", bpti_df), ("ood", ood_df)]:
        if args.core_col not in df.columns:
            raise ValueError(f"{name} missing --core-col: {args.core_col}")
        if args.fallback_col not in df.columns:
            raise ValueError(f"{name} missing --fallback-col: {args.fallback_col}")

    rows: list[dict] = []
    best_main = None
    lambda_grid = [-0.30, -0.20, -0.15, -0.10, -0.08, -0.05, -0.03, 0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    for lp, ls, shrink, thr in product(
        lambda_grid,
        lambda_grid,
        [0.00, 0.15, 0.30, 0.45, 0.60, 0.80],
        [0.50, 0.60, 0.70, 0.80],
    ):
        cfg = V16Config(
            name=f"v16_lp{lp:.2f}_ls{ls:.2f}_s{shrink:.2f}_t{thr:.2f}",
            core_col=args.core_col,
            lambda_phys2=float(lp),
            lambda_struct2=float(ls),
            shrink_scale=float(shrink),
            risk_thr=float(thr),
        )
        m = apply_v16_variant(main_df, cfg, fallback_col=args.fallback_col)
        m_main = eval_prediction(m, "rankscore_3view_v16")
        row = {
            "name": cfg.name,
            **cfg_to_dict(cfg),
            "main_rho": float(m_main["rho_new"]),
            "main_delta": float(m_main["delta"]),
            "main_gate": bool(float(m_main["rho_new"]) >= args.gate_main_rho),
            "main_shrink_mean": float(m["pred_shrink_v16"].mean()),
        }
        rows.append(row)
        if best_main is None or row["main_rho"] > best_main["main_rho"]:
            best_main = row

    variants_df = pd.DataFrame(rows).sort_values("main_rho", ascending=False)

    # Selection protocol: main-only tuning to avoid external leakage.
    chosen = best_main
    chosen_cfg = V16Config(
        name=str(chosen["name"]),
        core_col=str(chosen["core_col"]),
        lambda_phys2=float(chosen["lambda_phys2"]),
        lambda_struct2=float(chosen["lambda_struct2"]),
        shrink_scale=float(chosen["shrink_scale"]),
        risk_thr=float(chosen["risk_thr"]),
    )
    main_out = apply_v16_variant(main_df, chosen_cfg, fallback_col=args.fallback_col)
    bpti_out = apply_v16_variant(bpti_df, chosen_cfg, fallback_col=args.fallback_col)
    ood_out = apply_v16_variant(ood_df, chosen_cfg, fallback_col=args.fallback_col)

    chosen_triplet = _triplet_summary(
        main_out,
        bpti_out,
        ood_out,
        "rankscore_3view_v16",
        args.gate_main_rho,
        args.gate_external_delta,
    )

    # Post-hoc external report for all variants (not used for selection).
    ext_rows: list[dict] = []
    best_gate_posthoc = None
    for _, row in variants_df.iterrows():
        cfg = V16Config(
            name=str(row["name"]),
            core_col=str(row["core_col"]),
            lambda_phys2=float(row["lambda_phys2"]),
            lambda_struct2=float(row["lambda_struct2"]),
            shrink_scale=float(row["shrink_scale"]),
            risk_thr=float(row["risk_thr"]),
        )
        m = apply_v16_variant(main_df, cfg, fallback_col=args.fallback_col)
        b = apply_v16_variant(bpti_df, cfg, fallback_col=args.fallback_col)
        o = apply_v16_variant(ood_df, cfg, fallback_col=args.fallback_col)
        s = _triplet_summary(m, b, o, "rankscore_3view_v16", args.gate_main_rho, args.gate_external_delta)
        rec = {"name": cfg.name, **cfg_to_dict(cfg), **s}
        ext_rows.append(rec)
        if rec["gate_pass_all"]:
            if best_gate_posthoc is None or rec["main_rho"] > best_gate_posthoc["main_rho"]:
                best_gate_posthoc = rec
    ext_df = pd.DataFrame(ext_rows).sort_values("main_rho", ascending=False)

    refs = []
    for nm, col in [
        ("baseline_3view_base", "rankscore_3view_base"),
        ("target065", "rankscore_3view_target065"),
        ("v13_cons", args.core_col),
    ]:
        s = _triplet_summary(main_out, bpti_out, ood_out, col, args.gate_main_rho, args.gate_external_delta)
        refs.append({"candidate_type": "reference", "name": nm, **s})
    refs.append({"candidate_type": "variant", "name": chosen_cfg.name, **chosen_triplet})
    if best_gate_posthoc is not None:
        refs.append({"candidate_type": "variant_posthoc_gate", "name": best_gate_posthoc["name"], **best_gate_posthoc})
    refs_df = pd.DataFrame(refs)
    all_candidates = refs_df.copy()
    pareto = pareto_frontier_3d(all_candidates, "main_rho", "bpti_delta", "ood_delta")
    pareto = pareto.sort_values(["main_rho", "bpti_delta", "ood_delta"], ascending=[False, False, False])

    gate_json = {
        "gate_main_rho_threshold": float(args.gate_main_rho),
        "gate_external_delta_threshold": float(args.gate_external_delta),
        "selection_protocol": "main_only_tuning_then_single_external_evaluation",
        "enable_main_only_hist": bool(args.enable_main_only_hist),
        "chosen_variant": cfg_to_dict(chosen_cfg),
        "chosen_metrics": chosen_triplet,
        "best_main_variant": best_main,
        "best_gate_variant_posthoc": best_gate_posthoc,
        "n_variants": int(len(variants_df)),
        "n_gate_pass_variants_posthoc": int(ext_df["gate_pass_all"].sum()),
        "policy_pass": bool(policy["pass"]),
    }

    main_out.to_csv(out_dir / "main_eval_v16.csv", index=False)
    bpti_out.to_csv(out_dir / "bpti_eval_v16.csv", index=False)
    ood_out.to_csv(out_dir / "ood_eval_v16.csv", index=False)
    variants_df.to_csv(out_dir / "variant_metrics.csv", index=False)
    ext_df.to_csv(out_dir / "variant_external_metrics_posthoc.csv", index=False)
    all_candidates.to_csv(out_dir / "all_candidates_metrics.csv", index=False)
    pareto.to_csv(out_dir / "pareto_frontier.csv", index=False)
    (out_dir / "gate_metrics.json").write_text(json.dumps(gate_json, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "rules_config.json").write_text(
        json.dumps(
            {
                "core_col": args.core_col,
                "fallback_col": args.fallback_col,
                "chosen_variant": cfg_to_dict(chosen_cfg),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = [
        "# PepDDG v16 PS2+S2 (strict clean 3-channel)",
        "",
        f"- gate(main): rho>={args.gate_main_rho:.3f}",
        f"- gate(external): delta>={args.gate_external_delta:+.3f}",
        f"- policy(strict3): {'PASS' if policy['pass'] else 'FAIL'}",
        f"- searched variants (main-only tuning): {len(variants_df)}",
        f"- chosen gate-all: {'PASS' if bool(chosen_triplet['gate_pass_all']) else 'FAIL'}",
        f"- posthoc gate-pass variants: {int(ext_df['gate_pass_all'].sum())}",
        "",
        "## Chosen Variant",
        f"- {chosen_cfg.name}: main_rho={chosen['main_rho']:.6f}, "
        f"main_delta={chosen['main_delta']:+.6f}, "
        f"bpti_delta={chosen_triplet['bpti_delta']:+.6f}, "
        f"ood_delta={chosen_triplet['ood_delta']:+.6f}, "
        f"gate_all={'PASS' if bool(chosen_triplet['gate_pass_all']) else 'FAIL'}",
        "",
        "## Best Posthoc Gate Variant (Exploratory)",
    ]
    if best_gate_posthoc is not None:
        report.extend(
            [
                f"- {best_gate_posthoc['name']}: main_rho={best_gate_posthoc['main_rho']:.6f}, "
                f"bpti_delta={best_gate_posthoc['bpti_delta']:+.6f}, "
                f"ood_delta={best_gate_posthoc['ood_delta']:+.6f}, gate_all=PASS",
                "",
            ]
        )
    else:
        report.extend(["- None", ""])
    report.extend(
        [
        f"Outputs: `{out_dir}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print("=" * 72)
    print("PepDDG v16 PS2+S2 strict clean 3-channel")
    print("=" * 72)
    print(f"chosen:           {chosen_cfg.name}")
    print(f"main rho:         {chosen['main_rho']:.6f}")
    print(f"bpti delta:       {chosen_triplet['bpti_delta']:+.6f}")
    print(f"ood delta:        {chosen_triplet['ood_delta']:+.6f}")
    print(f"gate-pass count:  {int(ext_df['gate_pass_all'].sum())} (posthoc)")
    print(f"outputs:          {out_dir}")


if __name__ == "__main__":
    main()
