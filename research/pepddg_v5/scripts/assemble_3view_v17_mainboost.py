#!/usr/bin/env python
"""Assemble and evaluate PepDDG v17 main-rho boost (strict clean-3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.pepddg_v5.scripts.v14_ab_common import gate_summary  # noqa: E402
from research.pepddg_v5.scripts.v16_ps2s2_common import eval_prediction  # noqa: E402
from research.pepddg_v5.scripts.v17_mainboost_common import (  # noqa: E402
    V17Config,
    apply_v17_variant,
    cfg_to_dict,
    get_v17_used_columns,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-csv", default="research/pepddg_v5/results/v16_ps2s2/main_eval_v16.csv")
    parser.add_argument("--bpti-csv", default="research/pepddg_v5/results/v16_ps2s2/bpti_eval_v16.csv")
    parser.add_argument("--ood-csv", default="research/pepddg_v5/results/v16_ps2s2/ood_eval_v16.csv")
    parser.add_argument("--out-dir", default="research/pepddg_v5/results/v17_mainboost")
    parser.add_argument("--core-col", default="rankscore_3view_v16")
    parser.add_argument("--fallback-col", default="rankscore_3view_base")
    parser.add_argument("--gate-main-rho", type=float, default=0.680)
    parser.add_argument("--gate-external-delta", type=float, default=-0.010)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(args.main_csv)
    bpti_df = pd.read_csv(args.bpti_csv)
    ood_df = pd.read_csv(args.ood_csv)

    all_input_cols = sorted(set(main_df.columns) | set(bpti_df.columns) | set(ood_df.columns))
    used_cols = sorted(set(get_v17_used_columns() + [args.core_col, args.fallback_col]))
    policy = strict3_policy_audit(
        used_columns=used_cols,
        input_columns=all_input_cols,
    )
    (out_dir / "policy_audit.json").write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    if not policy["pass"]:
        raise ValueError(f"Strict3 policy failed: {policy['used_columns_violations']}")

    cfg = V17Config(core_col=args.core_col)
    main_out = apply_v17_variant(main_df, cfg)
    bpti_out = apply_v17_variant(bpti_df, cfg)
    ood_out = apply_v17_variant(ood_df, cfg)

    summary = _triplet_summary(
        main_out,
        bpti_out,
        ood_out,
        "rankscore_3view_v17_mainboost",
        args.gate_main_rho,
        args.gate_external_delta,
    )

    refs = []
    for nm, col in [
        ("baseline_3view_base", "rankscore_3view_base"),
        ("target065", "rankscore_3view_target065"),
        ("v16", args.core_col),
    ]:
        s = _triplet_summary(main_out, bpti_out, ood_out, col, args.gate_main_rho, args.gate_external_delta)
        refs.append({"candidate_type": "reference", "name": nm, **s})
    refs.append({"candidate_type": "variant", "name": cfg.name, **summary})
    refs_df = pd.DataFrame(refs)

    gate_json = {
        "gate_main_rho_threshold": float(args.gate_main_rho),
        "gate_external_delta_threshold": float(args.gate_external_delta),
        "selection_protocol": "fixed_deterministic_weights",
        "chosen_variant": cfg_to_dict(cfg),
        "chosen_metrics": summary,
        "policy_pass": bool(policy["pass"]),
    }

    main_out.to_csv(out_dir / "main_eval_v17.csv", index=False)
    bpti_out.to_csv(out_dir / "bpti_eval_v17.csv", index=False)
    ood_out.to_csv(out_dir / "ood_eval_v17.csv", index=False)
    refs_df.to_csv(out_dir / "all_candidates_metrics.csv", index=False)
    (out_dir / "gate_metrics.json").write_text(json.dumps(gate_json, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "rules_config.json").write_text(json.dumps(cfg_to_dict(cfg), indent=2, sort_keys=True), encoding="utf-8")

    report = [
        "# PepDDG v17 MainBoost (strict clean 3-channel)",
        "",
        f"- gate(main): rho>={args.gate_main_rho:.3f}",
        f"- gate(external): delta>={args.gate_external_delta:+.3f}",
        f"- policy(strict3): {'PASS' if policy['pass'] else 'FAIL'}",
        "",
        "## Fixed Variant",
        f"- {cfg.name}: main_rho={summary['main_rho']:.6f}, "
        f"main_delta={summary['main_delta']:+.6f}, "
        f"bpti_delta={summary['bpti_delta']:+.6f}, "
        f"ood_delta={summary['ood_delta']:+.6f}, "
        f"gate_all={'PASS' if bool(summary['gate_pass_all']) else 'FAIL'}",
        "",
        f"Outputs: `{out_dir}`",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print("=" * 72)
    print("PepDDG v17 MainBoost strict clean 3-channel")
    print("=" * 72)
    print(f"variant:          {cfg.name}")
    print(f"main rho:         {summary['main_rho']:.6f}")
    print(f"bpti delta:       {summary['bpti_delta']:+.6f}")
    print(f"ood delta:        {summary['ood_delta']:+.6f}")
    print(f"gate_all:         {'PASS' if bool(summary['gate_pass_all']) else 'FAIL'}")
    print(f"outputs:          {out_dir}")


if __name__ == "__main__":
    main()

