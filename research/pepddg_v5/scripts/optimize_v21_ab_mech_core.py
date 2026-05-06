#!/usr/bin/env python
"""PepDDG v21 A+B mechanism-core optimization (training-free, no new channels).

Protocol highlights:
- Uses only existing v19/v17.2/base experts and existing clean-3 columns.
- A-layer and B-layer tuning are main-only (external sets not used in objective).
- External metrics are evaluated after frozen candidate selection.
- Policy hard-fail on denylist hits (raw/loaded/used) or allowlist violations.
- Target-offset calibration is archived and disabled in v21 mainline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import rankdata


BANNED_PATTERNS = [
    r"foldx",
    r"rosetta",
    r"cartddg",
    r"stabddg",
    r"esm2",
    r"esmif",
    r"esm3",
    r"saprot",
    r"diffaffinity",
]


def _allowed_input_columns() -> set[str]:
    return {
        "target",
        "ddg_exp",
        "rankscore_3view_v19_strict3",
        "rankscore_3view_v17_2_oodguard",
        "rankscore_3view_base",
        "rankscore_phys",
        "rankscore_struct_base",
        "rankscore_mpnn",
        "ddg_paired_xint_iface",
        "ddg_xint_iface",
        "ddg_paired_bind_proxy",
        "ddg_bind_proxy",
        "n_iface_contacts_8a",
        "n_neighbors_10a",
        "burial_proxy_v10",
        "delta_volume",
        "delta_charge",
        "q_pack",
        "q_charge",
        "q_risk",
        "q_view_dispersion",
    }


def _used_columns() -> list[str]:
    return sorted(
        [
            "rankscore_3view_v19_strict3",
            "rankscore_3view_v17_2_oodguard",
            "rankscore_3view_base",
            "rankscore_phys",
            "rankscore_struct_base",
            "rankscore_mpnn",
            "ddg_xint_iface",
            "ddg_bind_proxy",
            "delta_volume",
            "delta_charge",
            "burial_proxy_v10",
            "q_pack",
            "q_charge",
            "q_risk",
            "q_view_dispersion",
            "n_iface_contacts_8a",
            "n_neighbors_10a",
        ]
    )


def _find_banned(columns: list[str], patterns: list[str] | None = None) -> list[str]:
    pats = BANNED_PATTERNS if patterns is None else patterns
    bad: list[str] = []
    for c in columns:
        cl = c.lower()
        if any(re.search(p, cl) for p in pats):
            bad.append(c)
    return sorted(set(bad))


def _outside_allowlist(columns: list[str], allowlist: set[str]) -> list[str]:
    return sorted(set([c for c in columns if c not in allowlist]))


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


def _rank_1d(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= 1:
        return np.zeros(n, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float) / float(n - 1)
    return ranks


def _metric_rank_1d(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= 1:
        return np.zeros(n, dtype=float)
    r = rankdata(v, method="average")
    return (r - 1.0) / float(n - 1)


def _rank_rows(scores_2d: np.ndarray) -> np.ndarray:
    s = np.asarray(scores_2d, dtype=float)
    n = s.shape[1]
    if n <= 1:
        return np.zeros_like(s, dtype=float)
    r = rankdata(s, axis=1, method="average")
    return (r - 1.0) / float(n - 1)


def _pearson_corr_rows(x: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    x0 = x - x.mean()
    y0 = y2d - y2d.mean(axis=1, keepdims=True)
    den = np.linalg.norm(x0) * np.linalg.norm(y0, axis=1)
    return (y0 @ x0) / np.maximum(den, 1e-12)


def _pearson_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.linalg.norm(x0) * np.linalg.norm(y0)
    return float((x0 @ y0) / max(float(den), 1e-12))


@dataclass
class DS:
    name: str
    df: pd.DataFrame
    input_columns: list[str]
    loaded_columns: list[str]
    y_rank: np.ndarray
    rho_base: float
    s19: np.ndarray
    s172: np.ndarray
    s0: np.ndarray
    z_unc_mat: np.ndarray
    z_chem_mat: np.ndarray
    z_risk: np.ndarray
    z_chem: np.ndarray
    z_attr: np.ndarray
    z_rep: np.ndarray
    z_geo: np.ndarray
    z_cons: np.ndarray
    z_mix: np.ndarray


def _resolve_phys_cols(df: pd.DataFrame) -> tuple[str, str]:
    p1 = "ddg_paired_xint_iface" if "ddg_paired_xint_iface" in df.columns else "ddg_xint_iface"
    p2 = "ddg_paired_bind_proxy" if "ddg_paired_bind_proxy" in df.columns else "ddg_bind_proxy"
    if p1 not in df.columns or p2 not in df.columns:
        raise ValueError("Missing physical proxy columns")
    return p1, p2


def _prepare(name: str, csv_path: Path) -> DS:
    raw_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    allow = _allowed_input_columns()
    df = pd.read_csv(csv_path, usecols=lambda c: c in allow)

    req = ["target", "ddg_exp", "rankscore_3view_v19_strict3", "rankscore_3view_v17_2_oodguard", "rankscore_3view_base"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    y_rank = _metric_rank_1d(_safe_numeric(df, "ddg_exp"))

    s19 = _rank_1d(_safe_numeric(df, "rankscore_3view_v19_strict3"))
    s172 = _rank_1d(_safe_numeric(df, "rankscore_3view_v17_2_oodguard"))
    s0 = _rank_1d(_safe_numeric(df, "rankscore_3view_base"))

    rho_base = _pearson_corr_1d(y_rank, _metric_rank_1d(_safe_numeric(df, "rankscore_3view_v19_strict3")))

    burial = np.clip(_safe_numeric(df, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    abs_vol = np.abs(_safe_numeric(df, "delta_volume", default=0.0))
    abs_chg = np.abs(_safe_numeric(df, "delta_charge", default=0.0))
    q_risk = _safe_numeric(df, "q_risk", default=0.5)
    q_disp = _safe_numeric(df, "q_view_dispersion", default=0.5)

    vol_bur_raw = abs_vol * burial
    chg_bur_raw = abs_chg * burial
    disp_risk_raw = q_disp * q_risk
    expert_shift_raw = np.abs(s19 - s172) + np.abs(s19 - s0)
    chem_risk_raw = 0.5 * vol_bur_raw + 0.5 * chg_bur_raw

    z_abs_vol = _rank_1d(abs_vol)
    z_abs_chg = _rank_1d(abs_chg)
    z_burial = _rank_1d(burial)
    z_qrisk = _rank_1d(q_risk)
    z_qdisp = _rank_1d(q_disp)
    z_expert_shift = _rank_1d(expert_shift_raw)
    z_vol_bur = _rank_1d(vol_bur_raw)
    z_chg_bur = _rank_1d(chg_bur_raw)
    z_disp_risk = _rank_1d(disp_risk_raw)
    z_chem = _rank_1d(chem_risk_raw)

    z_unc_mat = np.stack([z_qrisk, z_qdisp, z_expert_shift, z_abs_vol, z_abs_chg], axis=1)
    z_chem_mat = np.stack([z_vol_bur, z_chg_bur, z_disp_risk, z_burial], axis=1)
    z_risk = _rank_1d(
        0.35 * z_chem
        + 0.25 * z_disp_risk
        + 0.20 * z_expert_shift
        + 0.10 * z_qrisk
        + 0.10 * z_burial
    )

    p1, p2 = _resolve_phys_cols(df)
    z_phys = _rank_1d(_safe_numeric(df, "rankscore_phys", default=0.0))
    z_struct = _rank_1d(_safe_numeric(df, "rankscore_struct_base", default=0.0))
    z_mpnn = _rank_1d(_safe_numeric(df, "rankscore_mpnn", default=0.0))
    z_pack = _rank_1d(_safe_numeric(df, "q_pack", default=0.5))
    z_qcharge = _rank_1d(_safe_numeric(df, "q_charge", default=0.5))
    z_iface = _rank_1d(_safe_numeric(df, p1))
    z_bind = _rank_1d(_safe_numeric(df, p2))
    z_n_iface = _rank_1d(_safe_numeric(df, "n_iface_contacts_8a", default=0.0))
    z_n_neighbors = _rank_1d(_safe_numeric(df, "n_neighbors_10a", default=0.0))

    phys_mismatch_raw = np.abs(z_phys - 0.5 * (z_struct + z_mpnn))
    z_phys_mismatch = _rank_1d(phys_mismatch_raw)
    # Geometry strain proxy: disproportionate interface contacts vs local neighbor support.
    geo_strain_raw = np.maximum(0.0, z_n_iface - z_n_neighbors)
    z_geo = _rank_1d(geo_strain_raw)
    # Consensus instability proxy: expert disagreement amplified by phys mismatch.
    cons_instability_raw = z_expert_shift * z_phys_mismatch
    z_cons = _rank_1d(cons_instability_raw)
    # Interaction proxy between geometry strain and consensus instability.
    mix_instability_raw = geo_strain_raw * cons_instability_raw
    z_mix = _rank_1d(mix_instability_raw)

    attr_raw = (
        0.35 * z_mpnn
        + 0.20 * z_struct
        + 0.15 * z_pack
        + 0.10 * z_qcharge
        + 0.10 * z_iface
        + 0.10 * z_bind
    )
    rep_raw = (
        0.30 * z_phys_mismatch
        + 0.20 * z_disp_risk
        + 0.20 * z_vol_bur
        + 0.15 * z_chg_bur
        + 0.15 * z_expert_shift
    )
    z_attr = _rank_1d(attr_raw)
    z_rep = _rank_1d(rep_raw)

    return DS(
        name=name,
        df=df,
        input_columns=raw_cols,
        loaded_columns=list(df.columns),
        y_rank=y_rank,
        rho_base=rho_base,
        s19=s19,
        s172=s172,
        s0=s0,
        z_unc_mat=z_unc_mat,
        z_chem_mat=z_chem_mat,
        z_risk=z_risk,
        z_chem=z_chem,
        z_attr=z_attr,
        z_rep=z_rep,
        z_geo=z_geo,
        z_cons=z_cons,
        z_mix=z_mix,
    )


def _softmax2(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    den = np.maximum(ea + eb, 1e-12)
    return ea / den, eb / den


def _router_alpha(z: np.ndarray, w: np.ndarray, b0: np.ndarray, temp: np.ndarray) -> np.ndarray:
    logits = b0[:, None] + w @ z.T
    tt = np.clip(temp[:, None], 1e-6, None)
    return 1.0 / (1.0 + np.exp(-logits / tt))


def _target_groups(df: pd.DataFrame, min_size: int) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for _, idx in df.groupby("target").indices.items():
        ii = np.asarray(list(idx), dtype=int)
        if len(ii) >= int(min_size):
            groups.append(ii)
    return groups


def _target_rho_summary_rows(
    *,
    y_rank: np.ndarray,
    scores_rows: np.ndarray,
    target_groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n = scores_rows.shape[0]
    if not target_groups:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    vals = []
    for idx in target_groups:
        r = _rank_rows(scores_rows[:, idx])
        vals.append(_pearson_corr_rows(y_rank[idx], r))
    arr = np.stack(vals, axis=1)
    return np.median(arr, axis=1), np.std(arr, axis=1)


def _p95_abs_delta_rows(*, scores_rows: np.ndarray, ref_rank: np.ndarray) -> np.ndarray:
    r = _rank_rows(scores_rows)
    d = np.abs(r - ref_rank[None, :])
    return np.quantile(d, 0.95, axis=1)


def _robust_objective(
    *,
    main_rho: np.ndarray,
    scores_rows: np.ndarray,
    y_rank: np.ndarray,
    ref_rank: np.ndarray,
    target_groups: list[np.ndarray],
    lam_median: float,
    lam_std: float,
    lam_extreme: float,
    cap_p95: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    med_t, std_t = _target_rho_summary_rows(
        y_rank=y_rank,
        scores_rows=scores_rows,
        target_groups=target_groups,
    )
    p95 = _p95_abs_delta_rows(scores_rows=scores_rows, ref_rank=ref_rank)
    penalty = np.maximum(0.0, p95 - float(cap_p95))
    obj = main_rho + float(lam_median) * med_t - float(lam_std) * std_t - float(lam_extreme) * penalty
    return obj, {
        "median_target_rho": med_t,
        "std_target_rho": std_t,
        "p95_abs_delta": p95,
        "p95_penalty": penalty,
    }


def _b_dual_delta(
    *,
    alpha: np.ndarray,
    z_chem: np.ndarray,
    z_risk: np.ndarray,
    z_attr: np.ndarray,
    z_rep: np.ndarray,
    z_geo: np.ndarray,
    z_cons: np.ndarray,
    z_mix: np.ndarray,
    thr_unc: np.ndarray,
    thr_chem: np.ndarray,
    thr_risk: np.ndarray,
    lam_attr: np.ndarray,
    lam_rep: np.ndarray,
    lam_geo: np.ndarray,
    lam_cons: np.ndarray,
    lam_mix: np.ndarray,
    clip_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_attr = ((alpha <= thr_unc[:, None]) & (z_chem[None, :] <= thr_chem[:, None])).astype(float)
    mask_rep = (z_risk[None, :] >= thr_risk[:, None]).astype(float)
    delta = (
        lam_attr[:, None] * mask_attr * z_attr[None, :]
        - lam_rep[:, None] * mask_rep * z_rep[None, :]
        - lam_geo[:, None] * mask_rep * z_geo[None, :]
        - lam_cons[:, None] * mask_rep * z_cons[None, :]
        - lam_mix[:, None] * mask_rep * z_mix[None, :]
    )
    clip = np.abs(clip_b)[:, None]
    delta = np.clip(delta, -clip, clip)
    return delta, mask_attr, mask_rep


def _apply_trust_region(*, anchor: np.ndarray, cand_rows: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return anchor[None, :] + np.clip(cand_rows - anchor[None, :], -tau[:, None], tau[:, None])


def _eval_triplet(main_scores: np.ndarray, bpti_scores: np.ndarray, ood_scores: np.ndarray, dsm: DS, dsb: DS, dso: DS) -> dict[str, np.ndarray]:
    rm = _rank_rows(main_scores)
    rb = _rank_rows(bpti_scores)
    ro = _rank_rows(ood_scores)

    main_rho = _pearson_corr_rows(dsm.y_rank, rm)
    bpti_rho = _pearson_corr_rows(dsb.y_rank, rb)
    ood_rho = _pearson_corr_rows(dso.y_rank, ro)

    return {
        "main_rho": main_rho,
        "bpti_delta": bpti_rho - dsb.rho_base,
        "ood_delta": ood_rho - dso.rho_base,
    }


def _top_table(
    *,
    order_by: np.ndarray,
    metrics: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
    topk: int,
) -> pd.DataFrame:
    order = np.argsort(order_by)[::-1][: min(topk, len(order_by))]
    out = pd.DataFrame({"objective": order_by[order]})
    for k, v in metrics.items():
        out[k] = v[order]
    for k, v in params.items():
        out[k] = v[order]
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _policy_audit(datasets: list[DS]) -> dict:
    allow = _allowed_input_columns()
    used = _used_columns()

    raw_union = sorted(set(sum([ds.input_columns for ds in datasets], [])))
    loaded_union = sorted(set(sum([ds.loaded_columns for ds in datasets], [])))

    policy = {
        "allowlist": sorted(allow),
        "banned_patterns": BANNED_PATTERNS,
        "used_columns": used,
        "input_columns_raw": raw_union,
        "input_columns_loaded": loaded_union,
        "used_banned": _find_banned(used),
        "loaded_banned": _find_banned(loaded_union),
        "raw_banned": _find_banned(raw_union),
        "used_outside_allowlist": _outside_allowlist(used, allow),
        "loaded_outside_allowlist": _outside_allowlist(loaded_union, allow),
    }
    policy["pass"] = bool(
        len(policy["used_banned"]) == 0
        and len(policy["loaded_banned"]) == 0
        and len(policy["raw_banned"]) == 0
        and len(policy["used_outside_allowlist"]) == 0
        and len(policy["loaded_outside_allowlist"]) == 0
    )
    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/main_eval_v19.csv")
    ap.add_argument("--bpti-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/bpti_eval_v19.csv")
    ap.add_argument("--ood-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/ood_eval_v19.csv")
    ap.add_argument("--out-dir", default="research/pepddg_v5/results/v21_ab_mech_core")
    ap.add_argument("--seed", type=int, default=20260328)
    ap.add_argument("--n-a", type=int, default=80000)
    ap.add_argument("--n-b", type=int, default=80000)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--gate-main", type=float, default=0.75)
    ap.add_argument("--gate-delta", type=float, default=-0.01)

    ap.add_argument("--lam-attr-low", type=float, default=0.0)
    ap.add_argument("--lam-attr-high", type=float, default=2.0)
    ap.add_argument("--lam-rep-low", type=float, default=0.0)
    ap.add_argument("--lam-rep-high", type=float, default=2.0)
    ap.add_argument("--lam-geo-low", type=float, default=0.0)
    ap.add_argument("--lam-geo-high", type=float, default=0.0)
    ap.add_argument("--lam-cons-low", type=float, default=0.0)
    ap.add_argument("--lam-cons-high", type=float, default=0.0)
    ap.add_argument("--lam-mix-low", type=float, default=0.0)
    ap.add_argument("--lam-mix-high", type=float, default=0.0)

    ap.add_argument("--thr-unc-low", type=float, default=0.0)
    ap.add_argument("--thr-unc-high", type=float, default=0.95)
    ap.add_argument("--thr-chem-low", type=float, default=0.0)
    ap.add_argument("--thr-chem-high", type=float, default=0.95)
    ap.add_argument("--thr-risk-low", type=float, default=0.0)
    ap.add_argument("--thr-risk-high", type=float, default=0.95)

    ap.add_argument("--clip-low", type=float, default=0.02)
    ap.add_argument("--clip-high", type=float, default=0.25)
    ap.add_argument("--tau-low", type=float, default=0.05)
    ap.add_argument("--tau-high", type=float, default=0.40)
    ap.add_argument("--beta-safe-low", type=float, default=0.0)
    ap.add_argument("--beta-safe-high", type=float, default=0.40)

    ap.add_argument("--obj-lam-median", type=float, default=0.20)
    ap.add_argument("--obj-lam-std", type=float, default=0.10)
    ap.add_argument("--obj-lam-extreme", type=float, default=0.15)
    ap.add_argument("--obj-cap-p95-delta", type=float, default=0.35)
    ap.add_argument("--min-target-size", type=int, default=5)

    # Archived knob kept only for compatibility; v21 mainline hard-disables it.
    ap.add_argument("--enable-target-offset-b", type=int, choices=[0], default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dsm = _prepare("main", Path(args.main_csv))
    dsb = _prepare("bpti", Path(args.bpti_csv))
    dso = _prepare("ood", Path(args.ood_csv))

    policy = _policy_audit([dsm, dsb, dso])
    _write_json(out_dir / "policy_audit_v21_ab.json", policy)
    if not policy["pass"]:
        raise ValueError(f"policy failed: {policy}")

    if int(args.enable_target_offset_b) != 0:
        raise ValueError("v21 mainline forbids target-offset calibration")

    target_groups = _target_groups(dsm.df, min_size=int(args.min_target_size))

    rng = np.random.default_rng(int(args.seed))

    baseline = {
        "main": {
            "rho_base": float(dsm.rho_base),
            "n_mut": int(len(dsm.df)),
            "n_targets": int(dsm.df["target"].nunique()),
            "n_targets_obj": int(len(target_groups)),
        },
        "bpti": {"rho_base": float(dsb.rho_base), "n_mut": int(len(dsb.df)), "n_targets": int(dsb.df["target"].nunique())},
        "ood": {"rho_base": float(dso.rho_base), "n_mut": int(len(dso.df)), "n_targets": int(dso.df["target"].nunique())},
        "gates": {"main_rho": float(args.gate_main), "external_delta": float(args.gate_delta)},
        "objective": {
            "lam_median": float(args.obj_lam_median),
            "lam_std": float(args.obj_lam_std),
            "lam_extreme": float(args.obj_lam_extreme),
            "cap_p95_delta": float(args.obj_cap_p95_delta),
            "min_target_size": int(args.min_target_size),
        },
    }
    _write_json(out_dir / "baseline_snapshot_v21.json", baseline)

    # Stage A: dual-gate mechanism router (main-only robust objective)
    d_unc = dsm.z_unc_mat.shape[1]
    d_chem = dsm.z_chem_mat.shape[1]
    n_a = int(args.n_a)

    w_unc = rng.uniform(-2.0, 2.0, size=(n_a, d_unc))
    w_chem = rng.uniform(-2.0, 2.0, size=(n_a, d_chem))
    b_unc = rng.uniform(-2.0, 2.0, size=n_a)
    b_chem = rng.uniform(-2.0, 2.0, size=n_a)
    t_unc = rng.uniform(0.05, 0.9, size=n_a)
    t_chem = rng.uniform(0.05, 0.9, size=n_a)
    l172 = rng.uniform(-2.0, 2.0, size=n_a)
    l0 = rng.uniform(-2.0, 2.0, size=n_a)

    # Deterministic baseline candidate: alpha ~= 0 => v19 anchor.
    w_unc = np.vstack([np.zeros((1, d_unc), dtype=float), w_unc])
    w_chem = np.vstack([np.zeros((1, d_chem), dtype=float), w_chem])
    b_unc = np.concatenate([np.array([-20.0], dtype=float), b_unc])
    b_chem = np.concatenate([np.array([-20.0], dtype=float), b_chem])
    t_unc = np.concatenate([np.array([1.0], dtype=float), t_unc])
    t_chem = np.concatenate([np.array([1.0], dtype=float), t_chem])
    l172 = np.concatenate([np.array([0.0], dtype=float), l172])
    l0 = np.concatenate([np.array([0.0], dtype=float), l0])

    w172, w0 = _softmax2(l172, l0)

    g_unc_m = _router_alpha(dsm.z_unc_mat, w_unc, b_unc, t_unc)
    g_unc_b = _router_alpha(dsb.z_unc_mat, w_unc, b_unc, t_unc)
    g_unc_o = _router_alpha(dso.z_unc_mat, w_unc, b_unc, t_unc)

    g_chem_m = _router_alpha(dsm.z_chem_mat, w_chem, b_chem, t_chem)
    g_chem_b = _router_alpha(dsb.z_chem_mat, w_chem, b_chem, t_chem)
    g_chem_o = _router_alpha(dso.z_chem_mat, w_chem, b_chem, t_chem)

    alpha_m = np.clip(0.55 * g_unc_m + 0.45 * g_chem_m, 0.0, 1.0)
    alpha_b = np.clip(0.55 * g_unc_b + 0.45 * g_chem_b, 0.0, 1.0)
    alpha_o = np.clip(0.55 * g_unc_o + 0.45 * g_chem_o, 0.0, 1.0)

    safe_m = w172[:, None] * dsm.s172[None, :] + w0[:, None] * dsm.s0[None, :]
    safe_b = w172[:, None] * dsb.s172[None, :] + w0[:, None] * dsb.s0[None, :]
    safe_o = w172[:, None] * dso.s172[None, :] + w0[:, None] * dso.s0[None, :]

    sA_m = dsm.s19[None, :] + alpha_m * (safe_m - dsm.s19[None, :])
    sA_b = dsb.s19[None, :] + alpha_b * (safe_b - dsb.s19[None, :])
    sA_o = dso.s19[None, :] + alpha_o * (safe_o - dso.s19[None, :])

    metA = _eval_triplet(sA_m, sA_b, sA_o, dsm, dsb, dso)
    objA, compA = _robust_objective(
        main_rho=metA["main_rho"],
        scores_rows=sA_m,
        y_rank=dsm.y_rank,
        ref_rank=dsm.s19,
        target_groups=target_groups,
        lam_median=float(args.obj_lam_median),
        lam_std=float(args.obj_lam_std),
        lam_extreme=float(args.obj_lam_extreme),
        cap_p95=float(args.obj_cap_p95_delta),
    )
    idxA = int(np.argmax(objA))

    topA = _top_table(
        order_by=objA,
        metrics={
            "main_rho": metA["main_rho"],
            "bpti_delta": metA["bpti_delta"],
            "ood_delta": metA["ood_delta"],
            "median_target_rho": compA["median_target_rho"],
            "std_target_rho": compA["std_target_rho"],
            "p95_abs_delta": compA["p95_abs_delta"],
        },
        params={
            "b_unc": b_unc,
            "b_chem": b_chem,
            "t_unc": t_unc,
            "t_chem": t_chem,
            "w172": w172,
            "w0": w0,
            "w_unc_qrisk": w_unc[:, 0],
            "w_unc_qdisp": w_unc[:, 1],
            "w_unc_shift": w_unc[:, 2],
            "w_unc_vol": w_unc[:, 3],
            "w_unc_chg": w_unc[:, 4],
            "w_chem_volbur": w_chem[:, 0],
            "w_chem_chgbur": w_chem[:, 1],
            "w_chem_disp": w_chem[:, 2],
            "w_chem_burial": w_chem[:, 3],
        },
        topk=int(args.topk),
    )
    topA.to_csv(out_dir / "stageA_top.csv", index=False)

    selA = {
        "index": int(idxA),
        "strategy": "main_only_robust_objective",
        "params": {
            "b_unc": float(b_unc[idxA]),
            "b_chem": float(b_chem[idxA]),
            "t_unc": float(t_unc[idxA]),
            "t_chem": float(t_chem[idxA]),
            "w172": float(w172[idxA]),
            "w0": float(w0[idxA]),
            "w_unc": {
                "w_qrisk": float(w_unc[idxA, 0]),
                "w_qdisp": float(w_unc[idxA, 1]),
                "w_shift": float(w_unc[idxA, 2]),
                "w_vol": float(w_unc[idxA, 3]),
                "w_chg": float(w_unc[idxA, 4]),
            },
            "w_chem": {
                "w_volbur": float(w_chem[idxA, 0]),
                "w_chgbur": float(w_chem[idxA, 1]),
                "w_disp": float(w_chem[idxA, 2]),
                "w_burial": float(w_chem[idxA, 3]),
            },
        },
        "metrics": {
            "objective": float(objA[idxA]),
            "main_rho": float(metA["main_rho"][idxA]),
            "bpti_delta": float(metA["bpti_delta"][idxA]),
            "ood_delta": float(metA["ood_delta"][idxA]),
            "median_target_rho": float(compA["median_target_rho"][idxA]),
            "std_target_rho": float(compA["std_target_rho"][idxA]),
            "p95_abs_delta": float(compA["p95_abs_delta"][idxA]),
        },
    }
    _write_json(out_dir / "stageA_selected.json", selA)

    # Stage B: dual residual + trust region (main-only robust objective)
    alphaA_m = alpha_m[idxA]
    alphaA_b = alpha_b[idxA]
    alphaA_o = alpha_o[idxA]
    safeA_m = safe_m[idxA]
    safeA_b = safe_b[idxA]
    safeA_o = safe_o[idxA]
    sA_m_sel = sA_m[idxA]
    sA_b_sel = sA_b[idxA]
    sA_o_sel = sA_o[idxA]

    n_b = int(args.n_b)
    lam_attr = rng.uniform(float(args.lam_attr_low), float(args.lam_attr_high), size=n_b)
    lam_rep = rng.uniform(float(args.lam_rep_low), float(args.lam_rep_high), size=n_b)
    lam_geo = rng.uniform(float(args.lam_geo_low), float(args.lam_geo_high), size=n_b)
    lam_cons = rng.uniform(float(args.lam_cons_low), float(args.lam_cons_high), size=n_b)
    lam_mix = rng.uniform(float(args.lam_mix_low), float(args.lam_mix_high), size=n_b)
    thr_unc = rng.uniform(float(args.thr_unc_low), float(args.thr_unc_high), size=n_b)
    thr_chem = rng.uniform(float(args.thr_chem_low), float(args.thr_chem_high), size=n_b)
    thr_risk = rng.uniform(float(args.thr_risk_low), float(args.thr_risk_high), size=n_b)
    clip_b = rng.uniform(float(args.clip_low), float(args.clip_high), size=n_b)
    tau = rng.uniform(float(args.tau_low), float(args.tau_high), size=n_b)
    beta_safe = rng.uniform(float(args.beta_safe_low), float(args.beta_safe_high), size=n_b)

    # zero-correction candidate
    lam_attr = np.concatenate([np.array([0.0], dtype=float), lam_attr])
    lam_rep = np.concatenate([np.array([0.0], dtype=float), lam_rep])
    lam_geo = np.concatenate([np.array([0.0], dtype=float), lam_geo])
    lam_cons = np.concatenate([np.array([0.0], dtype=float), lam_cons])
    lam_mix = np.concatenate([np.array([0.0], dtype=float), lam_mix])
    thr_unc = np.concatenate([np.array([1.0], dtype=float), thr_unc])
    thr_chem = np.concatenate([np.array([1.0], dtype=float), thr_chem])
    thr_risk = np.concatenate([np.array([1.0], dtype=float), thr_risk])
    clip_b = np.concatenate([np.array([0.05], dtype=float), clip_b])
    tau = np.concatenate([np.array([0.40], dtype=float), tau])
    beta_safe = np.concatenate([np.array([0.0], dtype=float), beta_safe])

    alpha_rep_m = np.repeat(alphaA_m[None, :], len(lam_attr), axis=0)
    alpha_rep_b = np.repeat(alphaA_b[None, :], len(lam_attr), axis=0)
    alpha_rep_o = np.repeat(alphaA_o[None, :], len(lam_attr), axis=0)

    dB_m, mask_attr_m, mask_rep_m = _b_dual_delta(
        alpha=alpha_rep_m,
        z_chem=dsm.z_chem,
        z_risk=dsm.z_risk,
        z_attr=dsm.z_attr,
        z_rep=dsm.z_rep,
        z_geo=dsm.z_geo,
        z_cons=dsm.z_cons,
        z_mix=dsm.z_mix,
        thr_unc=thr_unc,
        thr_chem=thr_chem,
        thr_risk=thr_risk,
        lam_attr=lam_attr,
        lam_rep=lam_rep,
        lam_geo=lam_geo,
        lam_cons=lam_cons,
        lam_mix=lam_mix,
        clip_b=clip_b,
    )
    dB_b, mask_attr_b, mask_rep_b = _b_dual_delta(
        alpha=alpha_rep_b,
        z_chem=dsb.z_chem,
        z_risk=dsb.z_risk,
        z_attr=dsb.z_attr,
        z_rep=dsb.z_rep,
        z_geo=dsb.z_geo,
        z_cons=dsb.z_cons,
        z_mix=dsb.z_mix,
        thr_unc=thr_unc,
        thr_chem=thr_chem,
        thr_risk=thr_risk,
        lam_attr=lam_attr,
        lam_rep=lam_rep,
        lam_geo=lam_geo,
        lam_cons=lam_cons,
        lam_mix=lam_mix,
        clip_b=clip_b,
    )
    dB_o, mask_attr_o, mask_rep_o = _b_dual_delta(
        alpha=alpha_rep_o,
        z_chem=dso.z_chem,
        z_risk=dso.z_risk,
        z_attr=dso.z_attr,
        z_rep=dso.z_rep,
        z_geo=dso.z_geo,
        z_cons=dso.z_cons,
        z_mix=dso.z_mix,
        thr_unc=thr_unc,
        thr_chem=thr_chem,
        thr_risk=thr_risk,
        lam_attr=lam_attr,
        lam_rep=lam_rep,
        lam_geo=lam_geo,
        lam_cons=lam_cons,
        lam_mix=lam_mix,
        clip_b=clip_b,
    )

    sB_raw_m = sA_m_sel[None, :] + (1.0 - beta_safe[:, None]) * dB_m + beta_safe[:, None] * (safeA_m[None, :] - sA_m_sel[None, :])
    sB_raw_b = sA_b_sel[None, :] + (1.0 - beta_safe[:, None]) * dB_b + beta_safe[:, None] * (safeA_b[None, :] - sA_b_sel[None, :])
    sB_raw_o = sA_o_sel[None, :] + (1.0 - beta_safe[:, None]) * dB_o + beta_safe[:, None] * (safeA_o[None, :] - sA_o_sel[None, :])

    sB_m = _apply_trust_region(anchor=dsm.s19, cand_rows=sB_raw_m, tau=tau)
    sB_b = _apply_trust_region(anchor=dsb.s19, cand_rows=sB_raw_b, tau=tau)
    sB_o = _apply_trust_region(anchor=dso.s19, cand_rows=sB_raw_o, tau=tau)

    metB = _eval_triplet(sB_m, sB_b, sB_o, dsm, dsb, dso)
    objB, compB = _robust_objective(
        main_rho=metB["main_rho"],
        scores_rows=sB_m,
        y_rank=dsm.y_rank,
        ref_rank=dsm.s19,
        target_groups=target_groups,
        lam_median=float(args.obj_lam_median),
        lam_std=float(args.obj_lam_std),
        lam_extreme=float(args.obj_lam_extreme),
        cap_p95=float(args.obj_cap_p95_delta),
    )
    idxB = int(np.argmax(objB))

    topB = _top_table(
        order_by=objB,
        metrics={
            "main_rho": metB["main_rho"],
            "bpti_delta": metB["bpti_delta"],
            "ood_delta": metB["ood_delta"],
            "median_target_rho": compB["median_target_rho"],
            "std_target_rho": compB["std_target_rho"],
            "p95_abs_delta": compB["p95_abs_delta"],
            "mask_attr_frac_main": mask_attr_m.mean(axis=1),
            "mask_rep_frac_main": mask_rep_m.mean(axis=1),
        },
        params={
            "lam_attr": lam_attr,
            "lam_rep": lam_rep,
            "lam_geo": lam_geo,
            "lam_cons": lam_cons,
            "lam_mix": lam_mix,
            "thr_unc": thr_unc,
            "thr_chem": thr_chem,
            "thr_risk": thr_risk,
            "clip_b": clip_b,
            "tau": tau,
            "beta_safe": beta_safe,
        },
        topk=int(args.topk),
    )
    topB.to_csv(out_dir / "stageB_top.csv", index=False)

    final_main = sB_m[idxB]
    final_bpti = sB_b[idxB]
    final_ood = sB_o[idxB]

    if not (np.isfinite(final_main).all() and np.isfinite(final_bpti).all() and np.isfinite(final_ood).all()):
        raise ValueError("non-finite values found in final scores")

    selB = {
        "index": int(idxB),
        "strategy": "main_only_robust_objective",
        "params": {
            "lam_attr": float(lam_attr[idxB]),
            "lam_rep": float(lam_rep[idxB]),
            "lam_geo": float(lam_geo[idxB]),
            "lam_cons": float(lam_cons[idxB]),
            "lam_mix": float(lam_mix[idxB]),
            "thr_unc": float(thr_unc[idxB]),
            "thr_chem": float(thr_chem[idxB]),
            "thr_risk": float(thr_risk[idxB]),
            "clip_b": float(clip_b[idxB]),
            "tau": float(tau[idxB]),
            "beta_safe": float(beta_safe[idxB]),
        },
        "metrics": {
            "objective": float(objB[idxB]),
            "main_rho": float(metB["main_rho"][idxB]),
            "bpti_delta": float(metB["bpti_delta"][idxB]),
            "ood_delta": float(metB["ood_delta"][idxB]),
            "median_target_rho": float(compB["median_target_rho"][idxB]),
            "std_target_rho": float(compB["std_target_rho"][idxB]),
            "p95_abs_delta": float(compB["p95_abs_delta"][idxB]),
            "gate_main": bool(float(metB["main_rho"][idxB]) >= float(args.gate_main)),
            "gate_bpti": bool(float(metB["bpti_delta"][idxB]) >= float(args.gate_delta)),
            "gate_ood": bool(float(metB["ood_delta"][idxB]) >= float(args.gate_delta)),
        },
    }
    selB["metrics"]["gate_all"] = bool(selB["metrics"]["gate_main"] and selB["metrics"]["gate_bpti"] and selB["metrics"]["gate_ood"])
    _write_json(out_dir / "stageB_selected.json", selB)

    # Write final outputs
    out_main = dsm.df.copy()
    out_main["v21_a_alpha_unc"] = g_unc_m[idxA]
    out_main["v21_a_alpha_chem"] = g_chem_m[idxA]
    out_main["v21_a_alpha"] = alphaA_m
    out_main["v21_a_safe_score"] = _rank_1d(safeA_m)
    out_main["rankscore_3view_v21_a_route"] = _rank_1d(sA_m_sel)
    out_main["v21_b_mask_attr"] = mask_attr_m[idxB]
    out_main["v21_b_mask_rep"] = mask_rep_m[idxB]
    out_main["v21_b_local_delta"] = dB_m[idxB]
    out_main["rankscore_3view_v21_ab"] = _rank_1d(final_main)
    out_main.to_csv(out_dir / "main_eval_v21_ab.csv", index=False)

    out_bpti = dsb.df.copy()
    out_bpti["v21_a_alpha_unc"] = g_unc_b[idxA]
    out_bpti["v21_a_alpha_chem"] = g_chem_b[idxA]
    out_bpti["v21_a_alpha"] = alphaA_b
    out_bpti["v21_a_safe_score"] = _rank_1d(safeA_b)
    out_bpti["rankscore_3view_v21_a_route"] = _rank_1d(sA_b_sel)
    out_bpti["v21_b_mask_attr"] = mask_attr_b[idxB]
    out_bpti["v21_b_mask_rep"] = mask_rep_b[idxB]
    out_bpti["v21_b_local_delta"] = dB_b[idxB]
    out_bpti["rankscore_3view_v21_ab"] = _rank_1d(final_bpti)
    out_bpti.to_csv(out_dir / "bpti_eval_v21_ab.csv", index=False)

    out_ood = dso.df.copy()
    out_ood["v21_a_alpha_unc"] = g_unc_o[idxA]
    out_ood["v21_a_alpha_chem"] = g_chem_o[idxA]
    out_ood["v21_a_alpha"] = alphaA_o
    out_ood["v21_a_safe_score"] = _rank_1d(safeA_o)
    out_ood["rankscore_3view_v21_a_route"] = _rank_1d(sA_o_sel)
    out_ood["v21_b_mask_attr"] = mask_attr_o[idxB]
    out_ood["v21_b_mask_rep"] = mask_rep_o[idxB]
    out_ood["v21_b_local_delta"] = dB_o[idxB]
    out_ood["rankscore_3view_v21_ab"] = _rank_1d(final_ood)
    out_ood.to_csv(out_dir / "ood_eval_v21_ab.csv", index=False)

    summary = {
        "seed": int(args.seed),
        "gates": {"main_rho": float(args.gate_main), "external_delta": float(args.gate_delta)},
        "baseline": baseline,
        "stageA": selA,
        "stageB": selB,
    }
    _write_json(out_dir / "final_summary_v21_ab.json", summary)

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    cli_tail = [str(x) for x in sys.argv[1:]]
    command = " ".join(["python", shlex.quote(str(script_path)), *[shlex.quote(x) for x in cli_tail]])
    manifest = {
        "script": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "git_commit": _git_commit(repo_root),
        "argv": [str(x) for x in sys.argv],
        "command": command,
        "args": dict(vars(args)),
        "env": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy_version,
        },
        "inputs": {
            "main_csv": str(args.main_csv),
            "bpti_csv": str(args.bpti_csv),
            "ood_csv": str(args.ood_csv),
            "main_sha256": _sha256_file(Path(args.main_csv)),
            "bpti_sha256": _sha256_file(Path(args.bpti_csv)),
            "ood_sha256": _sha256_file(Path(args.ood_csv)),
            "main_rows": int(len(dsm.df)),
            "bpti_rows": int(len(dsb.df)),
            "ood_rows": int(len(dso.df)),
        },
    }
    _write_json(out_dir / "repro_manifest_v21_ab.json", manifest)

    rep = [
        "# PepDDG v21 A+B Mechanism Core (No New Channels)",
        "",
        f"- baseline main rho (v19): {dsm.rho_base:.6f}",
        f"- stageA objective: {selA['metrics']['objective']:.6f}",
        f"- stageA main rho: {selA['metrics']['main_rho']:.6f}",
        f"- stageB objective: {selB['metrics']['objective']:.6f}",
        f"- stageB main rho: {selB['metrics']['main_rho']:.6f}",
        f"- final bpti delta: {selB['metrics']['bpti_delta']:+.6f}",
        f"- final ood delta: {selB['metrics']['ood_delta']:+.6f}",
        f"- gate all: {selB['metrics']['gate_all']}",
        "",
        "## Objective",
        f"- pooled + {float(args.obj_lam_median):.3f}*median_target - {float(args.obj_lam_std):.3f}*std_target - {float(args.obj_lam_extreme):.3f}*max(0,p95-|delta|-{float(args.obj_cap_p95_delta):.3f})",
        "",
        "## Fixed Gates",
        f"- main >= {float(args.gate_main):.3f}",
        f"- external deltas >= {float(args.gate_delta):+.3f}",
    ]
    (out_dir / "report_v21_ab.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("=" * 72)
    print("PepDDG v21 A+B mechanism-core complete")
    print("=" * 72)
    print(f"main rho:   {selB['metrics']['main_rho']:.6f}")
    print(f"bpti delta: {selB['metrics']['bpti_delta']:+.6f}")
    print(f"ood delta:  {selB['metrics']['ood_delta']:+.6f}")
    print(f"gate all:   {selB['metrics']['gate_all']}")
    print(f"out dir:    {out_dir}")


if __name__ == "__main__":
    main()
