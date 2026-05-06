#!/usr/bin/env python
"""v19 strict clean-3 channelwise optimization (residual form, mainline anchored).

Route:
  Task0 baseline lock -> Task1 structure residual search -> Task2 physical
  residual search -> Task3 mpnn residual search -> Task4 minimal fusion.

Key difference from v1:
  - Starts from current mainline score as anchor when available.
  - Always includes zero-residual candidate (exact baseline point).
  - Uses residual parameterization to avoid uncontrolled distribution drift.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _rank_1d(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=float)
    return ranks.astype(float) / float(n - 1)


def _rank_rows(scores_2d: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks for metric computation."""
    s = np.asarray(scores_2d, dtype=float)
    n = s.shape[1]
    if n <= 1:
        return np.zeros_like(s, dtype=float)
    r = rankdata(s, axis=1, method="average")
    return (r - 1.0) / float(n - 1)


def _metric_rank_1d(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= 1:
        return np.zeros(n, dtype=float)
    r = rankdata(v, method="average")
    return (r - 1.0) / float(n - 1)


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


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


@dataclass(frozen=True)
class Spec:
    seed: int = 20260302
    gate_main: float = 0.700
    gate_delta: float = -0.010
    n_s: int = 50000
    n_p: int = 50000
    n_m: int = 50000
    n_f: int = 30000
    topk: int = 80


@dataclass
class DS:
    name: str
    df: pd.DataFrame
    input_columns: list[str]  # raw input header columns
    loaded_columns: list[str]  # columns actually loaded after allowlist filter
    y_rank: np.ndarray
    rho_base: float
    core: np.ndarray  # mainline anchor (or base fallback)
    phys_base: np.ndarray
    struct_base: np.ndarray
    mpnn_base: np.ndarray
    # structure primitives
    n_iface: np.ndarray
    n_neighbors: np.ndarray
    burial: np.ndarray
    vol_buried: np.ndarray
    charge_buried: np.ndarray
    # physical primitives
    xint: np.ndarray
    bind: np.ndarray
    q_pack: np.ndarray
    q_charge: np.ndarray
    # mpnn primitives
    mpnn_neg: np.ndarray
    mpnn_bind: np.ndarray
    mpnn_disp: np.ndarray


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


def _find_banned(columns: list[str], patterns: list[str] | None = None) -> list[str]:
    pats = BANNED_PATTERNS if patterns is None else patterns
    bad: list[str] = []
    for c in columns:
        cl = c.lower()
        if any(re.search(p, cl) for p in pats):
            bad.append(c)
    return sorted(set(bad))


def _allowed_input_columns(core_col: str | None = None) -> set[str]:
    allowed = {
        "target",
        "ddg_exp",
        "rankscore_phys",
        "rankscore_struct",
        "rankscore_struct_base",
        "rankscore_mpnn",
        "rankscore_3view_base",
        "rankscore_3view_v17_2_oodguard",
        "rankscore_3view_v17_mainboost",
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
        "mpnn_neg_llr_complex",
        "mpnn_ddg_bind",
    }
    if core_col:
        allowed.add(core_col)
    return allowed


def _resolve_phys_cols(df: pd.DataFrame) -> tuple[str, str]:
    c1 = "ddg_paired_xint_iface" if "ddg_paired_xint_iface" in df.columns else "ddg_xint_iface"
    c2 = "ddg_paired_bind_proxy" if "ddg_paired_bind_proxy" in df.columns else "ddg_bind_proxy"
    if c1 not in df.columns or c2 not in df.columns:
        raise ValueError("Missing physical proxy columns")
    return c1, c2


def _resolve_struct_col(df: pd.DataFrame) -> str:
    for c in ["rankscore_struct", "rankscore_struct_base", "struct_composite"]:
        if c in df.columns:
            return c
    raise ValueError("Missing struct column")


def _prepare(name: str, csv: Path, core_col: str | None = None) -> DS:
    raw_cols = list(pd.read_csv(csv, nrows=0).columns)
    allowed = _allowed_input_columns(core_col=core_col)
    df = pd.read_csv(csv, usecols=lambda c: c in allowed)
    y_rank = _metric_rank_1d(_safe_numeric(df, "ddg_exp"))

    p1, p2 = _resolve_phys_cols(df)
    phys_base = _rank_1d(_safe_numeric(df, "rankscore_phys")) if "rankscore_phys" in df.columns else _rank_1d(
        _rank_1d(_safe_numeric(df, p1)) + _rank_1d(_safe_numeric(df, p2))
    )
    sc = _resolve_struct_col(df)
    struct_base = _rank_1d(_safe_numeric(df, sc))
    if "rankscore_mpnn" in df.columns:
        mpnn_base = _rank_1d(_safe_numeric(df, "rankscore_mpnn"))
    else:
        mpnn_base = _rank_1d(_rank_1d(_safe_numeric(df, "mpnn_neg_llr_complex")) + _rank_1d(_safe_numeric(df, "mpnn_ddg_bind")))

    if core_col and core_col in df.columns:
        core_raw = _safe_numeric(df, core_col)
    elif "rankscore_3view_v17_2_oodguard" in df.columns:
        core_raw = _safe_numeric(df, "rankscore_3view_v17_2_oodguard")
    elif "rankscore_3view_v17_mainboost" in df.columns:
        core_raw = _safe_numeric(df, "rankscore_3view_v17_mainboost")
    elif "rankscore_3view_base" in df.columns:
        core_raw = _safe_numeric(df, "rankscore_3view_base")
    else:
        core_raw = phys_base + struct_base + mpnn_base
    core = _rank_1d(core_raw)
    rho_base = _pearson_corr_1d(y_rank, _metric_rank_1d(core_raw))

    burial = np.clip(_safe_numeric(df, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    abs_dv = np.abs(_safe_numeric(df, "delta_volume", default=0.0))
    abs_dc = np.abs(_safe_numeric(df, "delta_charge", default=0.0))

    mpnn_neg = _rank_1d(_safe_numeric(df, "mpnn_neg_llr_complex"))
    mpnn_bind = _rank_1d(_safe_numeric(df, "mpnn_ddg_bind"))

    return DS(
        name=name,
        df=df,
        input_columns=raw_cols,
        loaded_columns=list(df.columns),
        y_rank=y_rank,
        rho_base=rho_base,
        core=core,
        phys_base=phys_base,
        struct_base=struct_base,
        mpnn_base=mpnn_base,
        n_iface=_rank_1d(_safe_numeric(df, "n_iface_contacts_8a", default=0.0)),
        n_neighbors=_rank_1d(_safe_numeric(df, "n_neighbors_10a", default=0.0)),
        burial=_rank_1d(burial),
        vol_buried=_rank_1d(abs_dv * burial),
        charge_buried=_rank_1d(abs_dc * burial),
        xint=_rank_1d(_safe_numeric(df, p1)),
        bind=_rank_1d(_safe_numeric(df, p2)),
        q_pack=_rank_1d(_safe_numeric(df, "q_pack", default=0.5)),
        q_charge=_rank_1d(_safe_numeric(df, "q_charge", default=0.5)),
        mpnn_neg=mpnn_neg,
        mpnn_bind=mpnn_bind,
        mpnn_disp=_rank_1d(np.abs(mpnn_neg - mpnn_bind)),
    )


def _objective(main_rho: np.ndarray, baseline_main: float, bpti_delta: np.ndarray, ood_delta: np.ndarray, gate_delta: float) -> np.ndarray:
    pen_main_drop = 3.0 * np.maximum(0.0, baseline_main - main_rho)
    pen_bpti = 2.0 * np.maximum(0.0, gate_delta - bpti_delta)
    pen_ood = 1.2 * np.maximum(0.0, gate_delta - ood_delta)
    return main_rho - pen_main_drop - pen_bpti - pen_ood


def _eval(main_scores: np.ndarray, bpti_scores: np.ndarray, ood_scores: np.ndarray, dsm: DS, dsb: DS, dso: DS, gate_delta: float) -> dict[str, np.ndarray]:
    rm = _rank_rows(main_scores)
    rb = _rank_rows(bpti_scores)
    ro = _rank_rows(ood_scores)
    main_rho = _pearson_corr_rows(dsm.y_rank, rm)
    bpti_rho = _pearson_corr_rows(dsb.y_rank, rb)
    ood_rho = _pearson_corr_rows(dso.y_rank, ro)
    bpti_delta = bpti_rho - dsb.rho_base
    ood_delta = ood_rho - dso.rho_base
    obj = _objective(main_rho, dsm.rho_base, bpti_delta, ood_delta, gate_delta)
    return {"main_rho": main_rho, "bpti_delta": bpti_delta, "ood_delta": ood_delta, "objective": obj}


def _select(metrics: dict[str, np.ndarray], baseline_main: float, gate_main: float, gate_delta: float) -> tuple[int, str]:
    m = metrics["main_rho"]
    bd = metrics["bpti_delta"]
    od = metrics["ood_delta"]
    obj = metrics["objective"]
    mask70 = (m >= gate_main) & (bd >= gate_delta) & (od >= gate_delta)
    if np.any(mask70):
        ii = np.where(mask70)[0]
        return int(ii[np.argmax(m[ii])]), "gate70_max_main"
    mask_safe = (bd >= gate_delta) & (od >= gate_delta) & (m >= baseline_main - 0.002)
    if np.any(mask_safe):
        ii = np.where(mask_safe)[0]
        return int(ii[np.argmax(m[ii])]), "safe_nearbaseline_max_main"
    return int(np.argmax(obj)), "objective_fallback"


def _top(metrics: dict[str, np.ndarray], w: np.ndarray, cols: list[str], topk: int) -> pd.DataFrame:
    order = np.argsort(metrics["objective"])[::-1][: min(topk, len(w))]
    out = pd.DataFrame({
        "main_rho": metrics["main_rho"][order],
        "bpti_delta": metrics["bpti_delta"][order],
        "ood_delta": metrics["ood_delta"][order],
        "objective": metrics["objective"][order],
    })
    for i, c in enumerate(cols):
        out[c] = w[order, i]
    return out


def _sample_uniform_with_zero(rng: np.random.Generator, n: int, d: int, lo: float, hi: float) -> np.ndarray:
    w = rng.uniform(lo, hi, size=(n, d))
    z = np.zeros((1, d), dtype=float)
    return np.vstack([z, w])


def _struct_delta(ds: DS, w: np.ndarray) -> np.ndarray:
    # w: [iface, nei, burial, vol_buried, charge_buried]
    r = np.stack(
        [
            ds.n_iface - ds.struct_base,
            ds.n_neighbors - ds.struct_base,
            ds.burial - ds.struct_base,
            ds.vol_buried - ds.struct_base,
            ds.charge_buried - ds.struct_base,
        ],
        axis=1,
    )
    return w @ r.T


def _phys_delta(ds: DS, w: np.ndarray) -> np.ndarray:
    # w: [xint, bind, q_pack, q_charge, vol_buried, charge_buried]
    r = np.stack(
        [
            ds.xint - ds.phys_base,
            ds.bind - ds.phys_base,
            ds.q_pack - ds.phys_base,
            ds.q_charge - ds.phys_base,
            ds.vol_buried - ds.phys_base,
            ds.charge_buried - ds.phys_base,
        ],
        axis=1,
    )
    return w @ r.T


def _mpnn_delta(ds: DS, w: np.ndarray) -> np.ndarray:
    # w: [neg, bind, disp_penalty_nonneg]
    pos = w[:, [0]] * (ds.mpnn_neg - ds.mpnn_base)[None, :] + w[:, [1]] * (ds.mpnn_bind - ds.mpnn_base)[None, :]
    pen = np.abs(w[:, [2]]) * ds.mpnn_disp[None, :]
    return pos - pen


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/main_eval_v19.csv")
    ap.add_argument("--bpti-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/bpti_eval_v19.csv")
    ap.add_argument("--ood-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/ood_eval_v19.csv")
    ap.add_argument("--out-dir", default="research/pepddg_v5/results/v19_strict3_s2p2m2_v2")
    ap.add_argument("--seed", type=int, default=Spec.seed)
    ap.add_argument("--n-s", type=int, default=Spec.n_s)
    ap.add_argument("--n-p", type=int, default=Spec.n_p)
    ap.add_argument("--n-m", type=int, default=Spec.n_m)
    ap.add_argument("--n-f", type=int, default=Spec.n_f)
    ap.add_argument("--gate-main", type=float, default=Spec.gate_main)
    ap.add_argument("--gate-delta", type=float, default=Spec.gate_delta)
    ap.add_argument("--topk", type=int, default=Spec.topk)
    ap.add_argument("--core-col", default="rankscore_3view_v19_strict3")
    ap.add_argument("--range-s", type=float, default=1.2)
    ap.add_argument("--range-p", type=float, default=1.0)
    ap.add_argument("--range-m", type=float, default=1.0)
    ap.add_argument("--range-f-low", type=float, default=0.6)
    ap.add_argument("--range-f-high", type=float, default=1.4)
    args = ap.parse_args()

    spec = Spec(
        seed=int(args.seed),
        gate_main=float(args.gate_main),
        gate_delta=float(args.gate_delta),
        n_s=int(args.n_s),
        n_p=int(args.n_p),
        n_m=int(args.n_m),
        n_f=int(args.n_f),
        topk=int(args.topk),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)

    dsm = _prepare("main", Path(args.main_csv), core_col=str(args.core_col))
    dsb = _prepare("bpti", Path(args.bpti_csv), core_col=str(args.core_col))
    dso = _prepare("ood", Path(args.ood_csv), core_col=str(args.core_col))

    baseline = {
        "main": {"rho_base": float(dsm.rho_base), "n_mut": int(len(dsm.df)), "n_targets": int(dsm.df["target"].nunique())},
        "bpti": {"rho_base": float(dsb.rho_base), "n_mut": int(len(dsb.df)), "n_targets": int(dsb.df["target"].nunique())},
        "ood": {"rho_base": float(dso.rho_base), "n_mut": int(len(dso.df)), "n_targets": int(dso.df["target"].nunique())},
        "gates": {"main_rho": float(spec.gate_main), "external_delta": float(spec.gate_delta)},
    }
    _write_json(out_dir / "baseline_snapshot.json", baseline)

    # Fixed deltas accumulated stage-wise
    ds_m = np.zeros(len(dsm.df), dtype=float)
    dp_m = np.zeros(len(dsm.df), dtype=float)
    dm_m = np.zeros(len(dsm.df), dtype=float)
    ds_b = np.zeros(len(dsb.df), dtype=float)
    dp_b = np.zeros(len(dsb.df), dtype=float)
    dm_b = np.zeros(len(dsb.df), dtype=float)
    ds_o = np.zeros(len(dso.df), dtype=float)
    dp_o = np.zeros(len(dso.df), dtype=float)
    dm_o = np.zeros(len(dso.df), dtype=float)

    # Stage 1: structure residual search
    ws = _sample_uniform_with_zero(rng, spec.n_s, 5, -float(args.range_s), float(args.range_s))
    sdm = _struct_delta(dsm, ws)
    sdb = _struct_delta(dsb, ws)
    sdo = _struct_delta(dso, ws)
    met1 = _eval(dsm.core[None, :] + sdm, dsb.core[None, :] + sdb, dso.core[None, :] + sdo, dsm, dsb, dso, spec.gate_delta)
    i1, s1 = _select(met1, dsm.rho_base, spec.gate_main, spec.gate_delta)
    pd1 = _top(met1, ws, ["w_iface", "w_nei", "w_burial", "w_vol_buried", "w_charge_buried"], spec.topk)
    pd1.to_csv(out_dir / "stage1_structure_top.csv", index=False)
    sel1 = {
        "index": int(i1),
        "strategy": s1,
        "weights": {k: float(v) for k, v in zip(["w_iface", "w_nei", "w_burial", "w_vol_buried", "w_charge_buried"], ws[i1])},
        "metrics": {
            "main_rho": float(met1["main_rho"][i1]),
            "bpti_delta": float(met1["bpti_delta"][i1]),
            "ood_delta": float(met1["ood_delta"][i1]),
            "objective": float(met1["objective"][i1]),
        },
    }
    _write_json(out_dir / "stage1_structure_selected.json", sel1)
    ds_m, ds_b, ds_o = sdm[i1], sdb[i1], sdo[i1]

    # Stage 2: physical residual search
    wp = _sample_uniform_with_zero(rng, spec.n_p, 6, -float(args.range_p), float(args.range_p))
    pdm = _phys_delta(dsm, wp)
    pdb = _phys_delta(dsb, wp)
    pdo = _phys_delta(dso, wp)
    met2 = _eval(
        dsm.core[None, :] + ds_m[None, :] + pdm,
        dsb.core[None, :] + ds_b[None, :] + pdb,
        dso.core[None, :] + ds_o[None, :] + pdo,
        dsm,
        dsb,
        dso,
        spec.gate_delta,
    )
    i2, s2 = _select(met2, dsm.rho_base, spec.gate_main, spec.gate_delta)
    pd2 = _top(met2, wp, ["w_xint", "w_bind", "w_q_pack", "w_q_charge", "w_vol_buried", "w_charge_buried"], spec.topk)
    pd2.to_csv(out_dir / "stage2_physical_top.csv", index=False)
    sel2 = {
        "index": int(i2),
        "strategy": s2,
        "weights": {k: float(v) for k, v in zip(["w_xint", "w_bind", "w_q_pack", "w_q_charge", "w_vol_buried", "w_charge_buried"], wp[i2])},
        "metrics": {
            "main_rho": float(met2["main_rho"][i2]),
            "bpti_delta": float(met2["bpti_delta"][i2]),
            "ood_delta": float(met2["ood_delta"][i2]),
            "objective": float(met2["objective"][i2]),
        },
    }
    _write_json(out_dir / "stage2_physical_selected.json", sel2)
    dp_m, dp_b, dp_o = pdm[i2], pdb[i2], pdo[i2]

    # Stage 3: mpnn residual search
    wm = _sample_uniform_with_zero(rng, spec.n_m, 3, -float(args.range_m), float(args.range_m))
    mdm = _mpnn_delta(dsm, wm)
    mdb = _mpnn_delta(dsb, wm)
    mdo = _mpnn_delta(dso, wm)
    met3 = _eval(
        dsm.core[None, :] + ds_m[None, :] + dp_m[None, :] + mdm,
        dsb.core[None, :] + ds_b[None, :] + dp_b[None, :] + mdb,
        dso.core[None, :] + ds_o[None, :] + dp_o[None, :] + mdo,
        dsm,
        dsb,
        dso,
        spec.gate_delta,
    )
    i3, s3 = _select(met3, dsm.rho_base, spec.gate_main, spec.gate_delta)
    pd3 = _top(met3, wm, ["w_neg", "w_bind", "w_disp_pen"], spec.topk)
    pd3.to_csv(out_dir / "stage3_mpnn_top.csv", index=False)
    sel3 = {
        "index": int(i3),
        "strategy": s3,
        "weights": {k: float(v) for k, v in zip(["w_neg", "w_bind", "w_disp_pen"], wm[i3])},
        "metrics": {
            "main_rho": float(met3["main_rho"][i3]),
            "bpti_delta": float(met3["bpti_delta"][i3]),
            "ood_delta": float(met3["ood_delta"][i3]),
            "objective": float(met3["objective"][i3]),
        },
    }
    _write_json(out_dir / "stage3_mpnn_selected.json", sel3)
    dm_m, dm_b, dm_o = mdm[i3], mdb[i3], mdo[i3]

    # Stage 4: minimal fusion on residual amplitudes
    wf = _sample_uniform_with_zero(rng, spec.n_f, 3, float(args.range_f_low), float(args.range_f_high))
    wf[0, :] = np.array([1.0, 1.0, 1.0], dtype=float)
    sm = dsm.core[None, :] + wf[:, [0]] * ds_m[None, :] + wf[:, [1]] * dp_m[None, :] + wf[:, [2]] * dm_m[None, :]
    sb = dsb.core[None, :] + wf[:, [0]] * ds_b[None, :] + wf[:, [1]] * dp_b[None, :] + wf[:, [2]] * dm_b[None, :]
    so = dso.core[None, :] + wf[:, [0]] * ds_o[None, :] + wf[:, [1]] * dp_o[None, :] + wf[:, [2]] * dm_o[None, :]
    met4 = _eval(sm, sb, so, dsm, dsb, dso, spec.gate_delta)
    i4, s4 = _select(met4, dsm.rho_base, spec.gate_main, spec.gate_delta)
    pd4 = _top(met4, wf, ["w_s", "w_p", "w_m"], spec.topk)
    pd4.to_csv(out_dir / "stage4_fusion_top.csv", index=False)
    sel4 = {
        "index": int(i4),
        "strategy": s4,
        "weights": {k: float(v) for k, v in zip(["w_s", "w_p", "w_m"], wf[i4])},
        "metrics": {
            "main_rho": float(met4["main_rho"][i4]),
            "bpti_delta": float(met4["bpti_delta"][i4]),
            "ood_delta": float(met4["ood_delta"][i4]),
            "objective": float(met4["objective"][i4]),
            "gate_main": bool(float(met4["main_rho"][i4]) >= spec.gate_main),
            "gate_bpti": bool(float(met4["bpti_delta"][i4]) >= spec.gate_delta),
            "gate_ood": bool(float(met4["ood_delta"][i4]) >= spec.gate_delta),
        },
    }
    sel4["metrics"]["gate_all"] = bool(sel4["metrics"]["gate_main"] and sel4["metrics"]["gate_bpti"] and sel4["metrics"]["gate_ood"])
    _write_json(out_dir / "stage4_fusion_selected.json", sel4)

    # final per-row outputs
    w = wf[i4]
    final_main = dsm.core + w[0] * ds_m + w[1] * dp_m + w[2] * dm_m
    final_bpti = dsb.core + w[0] * ds_b + w[1] * dp_b + w[2] * dm_b
    final_ood = dso.core + w[0] * ds_o + w[1] * dp_o + w[2] * dm_o

    out_main = dsm.df.copy()
    out_main["v19_delta_struct"] = ds_m
    out_main["v19_delta_phys"] = dp_m
    out_main["v19_delta_mpnn"] = dm_m
    out_main["rankscore_3view_v19_strict3"] = _rank_1d(final_main)
    out_main.to_csv(out_dir / "main_eval_v19.csv", index=False)

    out_bpti = dsb.df.copy()
    out_bpti["v19_delta_struct"] = ds_b
    out_bpti["v19_delta_phys"] = dp_b
    out_bpti["v19_delta_mpnn"] = dm_b
    out_bpti["rankscore_3view_v19_strict3"] = _rank_1d(final_bpti)
    out_bpti.to_csv(out_dir / "bpti_eval_v19.csv", index=False)

    out_ood = dso.df.copy()
    out_ood["v19_delta_struct"] = ds_o
    out_ood["v19_delta_phys"] = dp_o
    out_ood["v19_delta_mpnn"] = dm_o
    out_ood["rankscore_3view_v19_strict3"] = _rank_1d(final_ood)
    out_ood.to_csv(out_dir / "ood_eval_v19.csv", index=False)

    used_columns = sorted(
        [
            "rankscore_struct",
            "rankscore_struct_base",
            "rankscore_phys",
            "rankscore_mpnn",
            "ddg_paired_xint_iface",
            "ddg_xint_iface",
            "ddg_paired_bind_proxy",
            "ddg_bind_proxy",
            "n_iface_contacts_8a",
            "n_neighbors_10a",
            "burial_proxy_v10",
            "delta_charge",
            "delta_volume",
            "q_pack",
            "q_charge",
            "mpnn_neg_llr_complex",
            "mpnn_ddg_bind",
        ]
    )
    input_columns_raw = sorted(set(dsm.input_columns + dsb.input_columns + dso.input_columns))
    input_columns_loaded = sorted(set(dsm.loaded_columns + dsb.loaded_columns + dso.loaded_columns))
    used_viol = _find_banned(used_columns)
    input_viol_raw = _find_banned(input_columns_raw)
    input_viol_loaded = _find_banned(input_columns_loaded)
    policy = {
        "strict_clean3_columns_used": used_columns,
        "input_columns_raw": input_columns_raw,
        "input_columns_loaded": input_columns_loaded,
        "banned_patterns": BANNED_PATTERNS,
        "used_columns_violations": used_viol,
        "input_columns_with_banned_patterns_raw": input_viol_raw,
        "input_columns_with_banned_patterns_loaded": input_viol_loaded,
        "pass": bool(len(used_viol) == 0 and len(input_viol_raw) == 0 and len(input_viol_loaded) == 0),
    }
    _write_json(out_dir / "policy_audit_v19.json", policy)

    summary = {
        "spec": asdict(spec),
        "core_col": str(args.core_col),
        "baseline": baseline,
        "stage1_structure": sel1,
        "stage2_physical": sel2,
        "stage3_mpnn": sel3,
        "stage4_fusion": sel4,
    }
    _write_json(out_dir / "final_summary_v19.json", summary)

    rep = [
        "# v19 strict clean-3 channelwise optimization (v2)",
        "",
        f"- baseline main rho: {dsm.rho_base:.6f}",
        f"- final main rho: {sel4['metrics']['main_rho']:.6f}",
        f"- final bpti delta: {sel4['metrics']['bpti_delta']:+.6f}",
        f"- final ood delta: {sel4['metrics']['ood_delta']:+.6f}",
        f"- gate all: {sel4['metrics']['gate_all']}",
    ]
    (out_dir / "report_v19.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("=" * 72)
    print("v19 strict clean-3 optimization (v2) complete")
    print("=" * 72)
    print(f"baseline_main = {dsm.rho_base:.6f}")
    print(f"main_rho      = {sel4['metrics']['main_rho']:.6f}")
    print(f"bpti_delta    = {sel4['metrics']['bpti_delta']:+.6f}")
    print(f"ood_delta     = {sel4['metrics']['ood_delta']:+.6f}")
    print(f"gate_all      = {sel4['metrics']['gate_all']}")
    print(f"out_dir       = {out_dir}")


if __name__ == "__main__":
    main()
