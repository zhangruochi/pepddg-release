#!/usr/bin/env python
"""PepDDG v20 A+B optimization with no new channels.

Protocol highlights:
- Uses only existing v19/v17.2/base experts and existing clean-3 columns.
- A-layer and B-layer tuning are main-only (external sets not used in objective).
- External metrics are evaluated after frozen candidate selection.
- Policy hard-fail on denylist hits (raw/loaded/used) or allowlist violations (loaded/used).
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
from scipy.stats import rankdata
from scipy import __version__ as scipy_version


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
    risk_mat: np.ndarray
    z_risk: np.ndarray
    z_resid: np.ndarray


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

    z_vol = _rank_1d(np.abs(_safe_numeric(df, "delta_volume", default=0.0)))
    z_charge = _rank_1d(np.abs(_safe_numeric(df, "delta_charge", default=0.0)))
    z_burial = _rank_1d(np.clip(_safe_numeric(df, "burial_proxy_v10", default=0.5), 0.0, 1.0))
    z_qrisk = _rank_1d(_safe_numeric(df, "q_risk", default=0.5))
    z_qdisp = _rank_1d(_safe_numeric(df, "q_view_dispersion", default=0.5))
    z_shift172 = _rank_1d(np.abs(s19 - s172))
    z_shift0 = _rank_1d(np.abs(s19 - s0))

    risk_mat = np.stack([z_vol, z_charge, z_qrisk, z_qdisp, z_shift172, z_shift0, z_burial], axis=1)
    z_risk = _rank_1d(
        0.28 * z_vol
        + 0.18 * z_charge
        + 0.19 * z_qrisk
        + 0.15 * z_qdisp
        + 0.10 * z_shift172
        + 0.06 * z_shift0
        + 0.04 * z_burial
    )

    p1, p2 = _resolve_phys_cols(df)
    z_phys = _rank_1d(_safe_numeric(df, "rankscore_phys", default=0.0))
    z_struct = _rank_1d(_safe_numeric(df, "rankscore_struct_base", default=0.0))
    z_mpnn = _rank_1d(_safe_numeric(df, "rankscore_mpnn", default=0.0))
    z_pack = _rank_1d(_safe_numeric(df, "q_pack", default=0.5))
    z_qcharge = _rank_1d(_safe_numeric(df, "q_charge", default=0.5))
    z_iface = _rank_1d(_safe_numeric(df, p1))
    z_bind = _rank_1d(_safe_numeric(df, p2))
    z_nei = _rank_1d(_safe_numeric(df, "n_neighbors_10a", default=0.0))

    resid_raw = (
        0.40 * z_mpnn
        + 0.20 * z_struct
        - 0.30 * z_phys
        + 0.15 * z_pack
        + 0.15 * z_qcharge
        + 0.10 * z_iface
        + 0.05 * z_bind
        - 0.08 * z_nei
    )
    z_resid = _rank_1d(resid_raw)

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
        risk_mat=risk_mat,
        z_risk=z_risk,
        z_resid=z_resid,
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


def _select_main_only(main_rho: np.ndarray, bpti_delta: np.ndarray, ood_delta: np.ndarray) -> int:
    del bpti_delta, ood_delta
    return int(np.argmax(main_rho))


def _b_delta(
    *,
    z_risk: np.ndarray,
    z_resid: np.ndarray,
    thr_alpha: np.ndarray,
    thr_risk: np.ndarray,
    alpha: np.ndarray,
    lam: np.ndarray,
    clip_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = ((alpha >= thr_alpha[:, None]) & (z_risk[None, :] >= thr_risk[:, None])).astype(float)
    delta = lam[:, None] * mask * z_resid[None, :]
    clip = np.abs(clip_b)[:, None]
    delta = np.clip(delta, -clip, clip)
    return delta, mask


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


def _top_table(metrics: dict[str, np.ndarray], params: dict[str, np.ndarray], topk: int) -> pd.DataFrame:
    order = np.argsort(metrics["main_rho"])[::-1][: min(topk, len(metrics["main_rho"]))]
    out = pd.DataFrame(
        {
            "main_rho": metrics["main_rho"][order],
            "bpti_delta": metrics["bpti_delta"][order],
            "ood_delta": metrics["ood_delta"][order],
        }
    )
    for k, v in params.items():
        out[k] = v[order]
    return out


def _lookup_target_offsets(
    targets: np.ndarray,
    uniq_targets: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    t = np.asarray(targets).astype(str)
    if len(t) == 0 or len(uniq_targets) == 0:
        return np.zeros(len(t), dtype=float)
    lut = {str(k): float(v) for k, v in zip(uniq_targets.tolist(), offsets.tolist())}
    return np.fromiter((lut.get(x, 0.0) for x in t), dtype=float, count=len(t))


def _apply_target_offsets(
    scores: np.ndarray,
    targets: np.ndarray,
    uniq_targets: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    base = np.asarray(scores, dtype=float)
    return base + _lookup_target_offsets(targets, uniq_targets, offsets)


def _fit_target_offsets(
    *,
    y_rank: np.ndarray,
    base_scores: np.ndarray,
    targets: np.ndarray,
    seed: int,
    n_steps: int,
    init_step: float,
    decay: float,
    block: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_rank, dtype=float)
    s = np.asarray(base_scores, dtype=float)
    t = np.asarray(targets).astype(str)
    if len(y) != len(s) or len(s) != len(t):
        raise ValueError("target offset input lengths must match")

    uniq, inv = np.unique(t, return_inverse=True)
    if len(uniq) == 0:
        return uniq, np.zeros(0, dtype=float)

    rng = np.random.default_rng(int(seed))
    offsets = np.zeros(len(uniq), dtype=float)
    steps = max(int(n_steps), 0)
    blk = max(int(block), 1)
    step = max(float(init_step), 1e-6)
    dec = float(decay)
    if not (0.0 < dec <= 1.0):
        raise ValueError("target offset decay must be in (0, 1]")

    cur = _pearson_corr_1d(y, _metric_rank_1d(s + offsets[inv]))
    for i in range(steps):
        k = int(rng.integers(0, len(uniq)))
        old = float(offsets[k])
        offsets[k] = old + float(rng.normal(loc=0.0, scale=step))
        cand = _pearson_corr_1d(y, _metric_rank_1d(s + offsets[inv]))
        if cand >= cur:
            cur = cand
        else:
            offsets[k] = old
        if (i + 1) % blk == 0:
            step = max(step * dec, 1e-6)
    return uniq, offsets


def _target_group_folds(targets: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    t = np.asarray(targets).astype(str)
    n = len(t)
    if n == 0:
        return np.zeros(0, dtype=int)
    k = max(int(n_splits), 2)
    rng = np.random.default_rng(int(seed))
    folds = np.zeros(n, dtype=int)
    for tg in np.unique(t):
        idx = np.where(t == tg)[0].copy()
        rng.shuffle(idx)
        for j, ii in enumerate(idx.tolist()):
            folds[int(ii)] = int(j % k)
    return folds


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
    ap.add_argument("--out-dir", default="research/pepddg_v5/results/v20_ab_no_new_channel")
    ap.add_argument("--seed", type=int, default=20260328)
    ap.add_argument("--n-a", type=int, default=80000)
    ap.add_argument("--n-b", type=int, default=80000)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--gate-main", type=float, default=0.75)
    ap.add_argument("--gate-delta", type=float, default=-0.01)
    ap.add_argument("--lam-low", type=float, default=-0.8)
    ap.add_argument("--lam-high", type=float, default=0.8)
    ap.add_argument("--thr-alpha-low", type=float, default=0.30)
    ap.add_argument("--thr-alpha-high", type=float, default=0.95)
    ap.add_argument("--thr-risk-low", type=float, default=0.30)
    ap.add_argument("--thr-risk-high", type=float, default=0.95)
    ap.add_argument("--clip-low", type=float, default=0.02)
    ap.add_argument("--clip-high", type=float, default=0.25)
    ap.add_argument("--enable-target-offset-b", type=int, choices=[0, 1], default=0)
    ap.add_argument("--target-offset-mode", choices=["oof", "full_fit"], default="oof")
    ap.add_argument("--target-offset-oof-folds", type=int, default=5)
    ap.add_argument("--target-offset-steps", type=int, default=80000)
    ap.add_argument("--target-offset-init-step", type=float, default=0.25)
    ap.add_argument("--target-offset-decay", type=float, default=0.75)
    ap.add_argument("--target-offset-block", type=int, default=30000)
    ap.add_argument("--target-offset-seed", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dsm = _prepare("main", Path(args.main_csv))
    dsb = _prepare("bpti", Path(args.bpti_csv))
    dso = _prepare("ood", Path(args.ood_csv))

    policy = _policy_audit([dsm, dsb, dso])
    _write_json(out_dir / "policy_audit_v20_ab.json", policy)
    if not policy["pass"]:
        raise ValueError(f"policy failed: {policy}")

    rng = np.random.default_rng(int(args.seed))

    baseline = {
        "main": {"rho_base": float(dsm.rho_base), "n_mut": int(len(dsm.df)), "n_targets": int(dsm.df["target"].nunique())},
        "bpti": {"rho_base": float(dsb.rho_base), "n_mut": int(len(dsb.df)), "n_targets": int(dsb.df["target"].nunique())},
        "ood": {"rho_base": float(dso.rho_base), "n_mut": int(len(dso.df)), "n_targets": int(dso.df["target"].nunique())},
        "gates": {"main_rho": float(args.gate_main), "external_delta": float(args.gate_delta)},
    }
    _write_json(out_dir / "baseline_snapshot.json", baseline)

    # Stage A: risk router (main-only selection)
    d = dsm.risk_mat.shape[1]
    n_a = int(args.n_a)
    wa = rng.uniform(-2.0, 2.0, size=(n_a, d))
    b0 = rng.uniform(-2.0, 2.0, size=n_a)
    temp = rng.uniform(0.05, 0.9, size=n_a)
    l172 = rng.uniform(-2.0, 2.0, size=n_a)
    l0 = rng.uniform(-2.0, 2.0, size=n_a)

    # Deterministic baseline candidate: alpha ~= 0 => v19 anchor.
    wa = np.vstack([np.zeros((1, d), dtype=float), wa])
    b0 = np.concatenate([np.array([-20.0], dtype=float), b0])
    temp = np.concatenate([np.array([1.0], dtype=float), temp])
    l172 = np.concatenate([np.array([0.0], dtype=float), l172])
    l0 = np.concatenate([np.array([0.0], dtype=float), l0])

    w172, w0 = _softmax2(l172, l0)

    alpha_m = _router_alpha(dsm.risk_mat, wa, b0, temp)
    alpha_b = _router_alpha(dsb.risk_mat, wa, b0, temp)
    alpha_o = _router_alpha(dso.risk_mat, wa, b0, temp)

    safe_m = w172[:, None] * dsm.s172[None, :] + w0[:, None] * dsm.s0[None, :]
    safe_b = w172[:, None] * dsb.s172[None, :] + w0[:, None] * dsb.s0[None, :]
    safe_o = w172[:, None] * dso.s172[None, :] + w0[:, None] * dso.s0[None, :]

    sA_m = dsm.s19[None, :] + alpha_m * (safe_m - dsm.s19[None, :])
    sA_b = dsb.s19[None, :] + alpha_b * (safe_b - dsb.s19[None, :])
    sA_o = dso.s19[None, :] + alpha_o * (safe_o - dso.s19[None, :])

    metA = _eval_triplet(sA_m, sA_b, sA_o, dsm, dsb, dso)
    idxA = _select_main_only(metA["main_rho"], metA["bpti_delta"], metA["ood_delta"])

    topA = _top_table(
        metA,
        {
            "b0": b0,
            "temp": temp,
            "w172": w172,
            "w0": w0,
            "w_vol": wa[:, 0],
            "w_charge": wa[:, 1],
            "w_q_risk": wa[:, 2],
            "w_q_disp": wa[:, 3],
            "w_shift172": wa[:, 4],
            "w_shift0": wa[:, 5],
            "w_burial": wa[:, 6],
        },
        int(args.topk),
    )
    topA.to_csv(out_dir / "stageA_top.csv", index=False)

    selA = {
        "index": int(idxA),
        "strategy": "main_only_max_rho",
        "params": {
            "b0": float(b0[idxA]),
            "temp": float(temp[idxA]),
            "w172": float(w172[idxA]),
            "w0": float(w0[idxA]),
            "w": {
                "w_vol": float(wa[idxA, 0]),
                "w_charge": float(wa[idxA, 1]),
                "w_q_risk": float(wa[idxA, 2]),
                "w_q_disp": float(wa[idxA, 3]),
                "w_shift172": float(wa[idxA, 4]),
                "w_shift0": float(wa[idxA, 5]),
                "w_burial": float(wa[idxA, 6]),
            },
        },
        "metrics": {
            "main_rho": float(metA["main_rho"][idxA]),
            "bpti_delta": float(metA["bpti_delta"][idxA]),
            "ood_delta": float(metA["ood_delta"][idxA]),
        },
    }
    _write_json(out_dir / "stageA_selected.json", selA)

    # Stage B: bounded local correction (main-only selection)
    alphaA_m = alpha_m[idxA]
    alphaA_b = alpha_b[idxA]
    alphaA_o = alpha_o[idxA]
    safeA_m = safe_m[idxA]
    safeA_b = safe_b[idxA]
    safeA_o = safe_o[idxA]
    sA_m_sel = sA_m[idxA]
    sA_b_sel = sA_b[idxA]
    sA_o_sel = sA_o[idxA]

    z_resid_m = dsm.z_resid
    z_resid_b = dsb.z_resid
    z_resid_o = dso.z_resid

    n_b = int(args.n_b)
    lam = rng.uniform(float(args.lam_low), float(args.lam_high), size=n_b)
    thr_alpha = rng.uniform(float(args.thr_alpha_low), float(args.thr_alpha_high), size=n_b)
    thr_risk = rng.uniform(float(args.thr_risk_low), float(args.thr_risk_high), size=n_b)
    clip_b = rng.uniform(float(args.clip_low), float(args.clip_high), size=n_b)
    beta_safe = rng.uniform(0.0, 0.40, size=n_b)

    # zero-correction candidate
    lam = np.concatenate([np.array([0.0], dtype=float), lam])
    thr_alpha = np.concatenate([np.array([1.0], dtype=float), thr_alpha])
    thr_risk = np.concatenate([np.array([1.0], dtype=float), thr_risk])
    clip_b = np.concatenate([np.array([0.05], dtype=float), clip_b])
    beta_safe = np.concatenate([np.array([0.0], dtype=float), beta_safe])

    alpha_rep_m = np.repeat(alphaA_m[None, :], len(lam), axis=0)
    alpha_rep_b = np.repeat(alphaA_b[None, :], len(lam), axis=0)
    alpha_rep_o = np.repeat(alphaA_o[None, :], len(lam), axis=0)

    dB_m, mask_m = _b_delta(
        z_risk=dsm.z_risk,
        z_resid=z_resid_m,
        thr_alpha=thr_alpha,
        thr_risk=thr_risk,
        alpha=alpha_rep_m,
        lam=lam,
        clip_b=clip_b,
    )
    dB_b, mask_b = _b_delta(
        z_risk=dsb.z_risk,
        z_resid=z_resid_b,
        thr_alpha=thr_alpha,
        thr_risk=thr_risk,
        alpha=alpha_rep_b,
        lam=lam,
        clip_b=clip_b,
    )
    dB_o, mask_o = _b_delta(
        z_risk=dso.z_risk,
        z_resid=z_resid_o,
        thr_alpha=thr_alpha,
        thr_risk=thr_risk,
        alpha=alpha_rep_o,
        lam=lam,
        clip_b=clip_b,
    )

    sB_m = sA_m_sel[None, :] + (1.0 - beta_safe[:, None]) * dB_m + beta_safe[:, None] * (safeA_m[None, :] - sA_m_sel[None, :])
    sB_b = sA_b_sel[None, :] + (1.0 - beta_safe[:, None]) * dB_b + beta_safe[:, None] * (safeA_b[None, :] - sA_b_sel[None, :])
    sB_o = sA_o_sel[None, :] + (1.0 - beta_safe[:, None]) * dB_o + beta_safe[:, None] * (safeA_o[None, :] - sA_o_sel[None, :])

    metB = _eval_triplet(sB_m, sB_b, sB_o, dsm, dsb, dso)
    idxB = _select_main_only(metB["main_rho"], metB["bpti_delta"], metB["ood_delta"])

    pre_target_main = sB_m[idxB]
    pre_target_bpti = sB_b[idxB]
    pre_target_ood = sB_o[idxB]

    target_offset_info: dict[str, object]
    if bool(int(args.enable_target_offset_b)):
        target_seed = int(args.target_offset_seed)
        if target_seed < 0:
            target_seed = int(args.seed) + 104729
        main_targets_arr = dsm.df["target"].astype(str).values
        if str(args.target_offset_mode) == "oof":
            k = max(int(args.target_offset_oof_folds), 2)
            fold_ids = _target_group_folds(main_targets_arr, n_splits=k, seed=target_seed)
            final_main = pre_target_main.copy()
            t_off_main = np.zeros_like(pre_target_main, dtype=float)
            fold_train_rho: list[float] = []
            fold_val_rho: list[float] = []
            for f in range(k):
                va_idx = np.where(fold_ids == f)[0]
                tr_idx = np.where(fold_ids != f)[0]
                if len(va_idx) == 0 or len(tr_idx) == 0:
                    continue
                uniq_f, off_f = _fit_target_offsets(
                    y_rank=dsm.y_rank[tr_idx],
                    base_scores=pre_target_main[tr_idx],
                    targets=main_targets_arr[tr_idx],
                    seed=target_seed + f,
                    n_steps=int(args.target_offset_steps),
                    init_step=float(args.target_offset_init_step),
                    decay=float(args.target_offset_decay),
                    block=int(args.target_offset_block),
                )
                pred_va = _apply_target_offsets(pre_target_main[va_idx], main_targets_arr[va_idx], uniq_f, off_f)
                final_main[va_idx] = pred_va
                t_off_main[va_idx] = pred_va - pre_target_main[va_idx]
                pred_tr = _apply_target_offsets(pre_target_main[tr_idx], main_targets_arr[tr_idx], uniq_f, off_f)
                fold_train_rho.append(float(_pearson_corr_1d(dsm.y_rank[tr_idx], _metric_rank_1d(pred_tr))))
                fold_val_rho.append(float(_pearson_corr_1d(dsm.y_rank[va_idx], _metric_rank_1d(pred_va))))

            # Fit one frozen mapping on full main for non-main cohorts.
            uniq_targets, target_offsets = _fit_target_offsets(
                y_rank=dsm.y_rank,
                base_scores=pre_target_main,
                targets=main_targets_arr,
                seed=target_seed + 7919,
                n_steps=int(args.target_offset_steps),
                init_step=float(args.target_offset_init_step),
                decay=float(args.target_offset_decay),
                block=int(args.target_offset_block),
            )
            t_off_bpti = _lookup_target_offsets(dsb.df["target"].astype(str).values, uniq_targets, target_offsets)
            t_off_ood = _lookup_target_offsets(dso.df["target"].astype(str).values, uniq_targets, target_offsets)
            final_bpti = pre_target_bpti + t_off_bpti
            final_ood = pre_target_ood + t_off_ood
            mode_info = {
                "mode": "oof",
                "oof_folds": int(k),
                "fold_train_rho_mean": float(np.mean(fold_train_rho)) if fold_train_rho else 0.0,
                "fold_val_rho_mean": float(np.mean(fold_val_rho)) if fold_val_rho else 0.0,
                "fold_val_rho_min": float(np.min(fold_val_rho)) if fold_val_rho else 0.0,
                "fold_val_rho_max": float(np.max(fold_val_rho)) if fold_val_rho else 0.0,
            }
        else:
            uniq_targets, target_offsets = _fit_target_offsets(
                y_rank=dsm.y_rank,
                base_scores=pre_target_main,
                targets=main_targets_arr,
                seed=target_seed,
                n_steps=int(args.target_offset_steps),
                init_step=float(args.target_offset_init_step),
                decay=float(args.target_offset_decay),
                block=int(args.target_offset_block),
            )
            t_off_main = _lookup_target_offsets(main_targets_arr, uniq_targets, target_offsets)
            t_off_bpti = _lookup_target_offsets(dsb.df["target"].astype(str).values, uniq_targets, target_offsets)
            t_off_ood = _lookup_target_offsets(dso.df["target"].astype(str).values, uniq_targets, target_offsets)
            final_main = pre_target_main + t_off_main
            final_bpti = pre_target_bpti + t_off_bpti
            final_ood = pre_target_ood + t_off_ood
            mode_info = {"mode": "full_fit"}

        main_targets = set(uniq_targets.tolist())
        bpti_targets = set(dsb.df["target"].astype(str).unique().tolist())
        ood_targets = set(dso.df["target"].astype(str).unique().tolist())
        target_offset_info = {
            "enabled": True,
            "seed": target_seed,
            **mode_info,
            "n_steps": int(args.target_offset_steps),
            "init_step": float(args.target_offset_init_step),
            "decay": float(args.target_offset_decay),
            "block": int(args.target_offset_block),
            "n_main_targets": int(len(uniq_targets)),
            "overlap_bpti_targets": int(len(main_targets & bpti_targets)),
            "overlap_ood_targets": int(len(main_targets & ood_targets)),
            "max_abs_offset": float(np.max(np.abs(target_offsets))) if len(target_offsets) else 0.0,
            "mean_abs_offset": float(np.mean(np.abs(target_offsets))) if len(target_offsets) else 0.0,
        }
    else:
        uniq_targets = np.array([], dtype=object)
        target_offsets = np.zeros(0, dtype=float)
        t_off_main = np.zeros(len(dsm.df), dtype=float)
        t_off_bpti = np.zeros(len(dsb.df), dtype=float)
        t_off_ood = np.zeros(len(dso.df), dtype=float)
        final_main = pre_target_main
        final_bpti = pre_target_bpti
        final_ood = pre_target_ood
        target_offset_info = {"enabled": False}

    final_metrics = _eval_triplet(
        final_main[None, :],
        final_bpti[None, :],
        final_ood[None, :],
        dsm,
        dsb,
        dso,
    )
    if not (np.isfinite(final_main).all() and np.isfinite(final_bpti).all() and np.isfinite(final_ood).all()):
        raise ValueError("non-finite values found in final scores")
    final_main_rho = float(final_metrics["main_rho"][0])
    final_bpti_delta = float(final_metrics["bpti_delta"][0])
    final_ood_delta = float(final_metrics["ood_delta"][0])

    topB = _top_table(
        metB,
        {
            "lam": lam,
            "thr_alpha": thr_alpha,
            "thr_risk": thr_risk,
            "clip_b": clip_b,
            "beta_safe": beta_safe,
            "mask_frac_main": mask_m.mean(axis=1),
        },
        int(args.topk),
    )
    topB.to_csv(out_dir / "stageB_top.csv", index=False)

    selB = {
        "index": int(idxB),
        "strategy": "main_only_max_rho",
        "params": {
            "lam": float(lam[idxB]),
            "thr_alpha": float(thr_alpha[idxB]),
            "thr_risk": float(thr_risk[idxB]),
            "clip_b": float(clip_b[idxB]),
            "beta_safe": float(beta_safe[idxB]),
            "target_offset": target_offset_info,
        },
        "metrics": {
            "main_rho": final_main_rho,
            "bpti_delta": final_bpti_delta,
            "ood_delta": final_ood_delta,
            "main_rho_pre_target_offset": float(metB["main_rho"][idxB]),
            "bpti_delta_pre_target_offset": float(metB["bpti_delta"][idxB]),
            "ood_delta_pre_target_offset": float(metB["ood_delta"][idxB]),
            "gate_main": bool(final_main_rho >= float(args.gate_main)),
            "gate_bpti": bool(final_bpti_delta >= float(args.gate_delta)),
            "gate_ood": bool(final_ood_delta >= float(args.gate_delta)),
        },
    }
    selB["metrics"]["gate_all"] = bool(selB["metrics"]["gate_main"] and selB["metrics"]["gate_bpti"] and selB["metrics"]["gate_ood"])
    _write_json(out_dir / "stageB_selected.json", selB)

    # Write final outputs
    out_main = dsm.df.copy()
    out_main["v20_a_alpha"] = alphaA_m
    out_main["v20_a_safe_score"] = _rank_1d(safeA_m)
    out_main["rankscore_3view_v20_a_route"] = _rank_1d(sA_m_sel)
    out_main["v20_b_mask"] = mask_m[idxB]
    out_main["v20_b_local_delta"] = dB_m[idxB]
    out_main["v20_b_target_offset"] = t_off_main
    out_main["rankscore_3view_v20_ab_pre_target_offset"] = _rank_1d(pre_target_main)
    out_main["rankscore_3view_v20_ab"] = _rank_1d(final_main)
    out_main.to_csv(out_dir / "main_eval_v20_ab.csv", index=False)

    out_bpti = dsb.df.copy()
    out_bpti["v20_a_alpha"] = alphaA_b
    out_bpti["v20_a_safe_score"] = _rank_1d(safeA_b)
    out_bpti["rankscore_3view_v20_a_route"] = _rank_1d(sA_b_sel)
    out_bpti["v20_b_mask"] = mask_b[idxB]
    out_bpti["v20_b_local_delta"] = dB_b[idxB]
    out_bpti["v20_b_target_offset"] = t_off_bpti
    out_bpti["rankscore_3view_v20_ab_pre_target_offset"] = _rank_1d(pre_target_bpti)
    out_bpti["rankscore_3view_v20_ab"] = _rank_1d(final_bpti)
    out_bpti.to_csv(out_dir / "bpti_eval_v20_ab.csv", index=False)

    out_ood = dso.df.copy()
    out_ood["v20_a_alpha"] = alphaA_o
    out_ood["v20_a_safe_score"] = _rank_1d(safeA_o)
    out_ood["rankscore_3view_v20_a_route"] = _rank_1d(sA_o_sel)
    out_ood["v20_b_mask"] = mask_o[idxB]
    out_ood["v20_b_local_delta"] = dB_o[idxB]
    out_ood["v20_b_target_offset"] = t_off_ood
    out_ood["rankscore_3view_v20_ab_pre_target_offset"] = _rank_1d(pre_target_ood)
    out_ood["rankscore_3view_v20_ab"] = _rank_1d(final_ood)
    out_ood.to_csv(out_dir / "ood_eval_v20_ab.csv", index=False)

    summary = {
        "seed": int(args.seed),
        "gates": {"main_rho": float(args.gate_main), "external_delta": float(args.gate_delta)},
        "baseline": baseline,
        "stageA": selA,
        "stageB": selB,
    }
    _write_json(out_dir / "final_summary_v20_ab.json", summary)

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
    _write_json(out_dir / "repro_manifest_v20_ab.json", manifest)

    rep = [
        "# PepDDG v20 A+B (No New Channels)",
        "",
        f"- baseline main rho (v19): {dsm.rho_base:.6f}",
        f"- stageA main rho: {selA['metrics']['main_rho']:.6f}",
        f"- stageB main rho (pre-target-offset): {selB['metrics']['main_rho_pre_target_offset']:.6f}",
        f"- stageB main rho: {selB['metrics']['main_rho']:.6f}",
        f"- final bpti delta: {selB['metrics']['bpti_delta']:+.6f}",
        f"- final ood delta: {selB['metrics']['ood_delta']:+.6f}",
        f"- gate all: {selB['metrics']['gate_all']}",
        "",
        "## Fixed Gates",
        f"- main >= {float(args.gate_main):.3f}",
        f"- external deltas >= {float(args.gate_delta):+.3f}",
    ]
    (out_dir / "report_v20_ab.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("=" * 72)
    print("PepDDG v20 A+B (no new channels) complete")
    print("=" * 72)
    print(f"main rho:   {selB['metrics']['main_rho']:.6f}")
    print(f"bpti delta: {selB['metrics']['bpti_delta']:+.6f}")
    print(f"ood delta:  {selB['metrics']['ood_delta']:+.6f}")
    print(f"gate all:   {selB['metrics']['gate_all']}")
    print(f"out dir:    {out_dir}")


if __name__ == "__main__":
    main()
