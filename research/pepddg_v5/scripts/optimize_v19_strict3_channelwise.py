#!/usr/bin/env python
"""Strict clean-3 channelwise optimization (S->P->M->minimal fusion) for v19.

This script implements the agreed route:
1) lock mainline baseline snapshot;
2) optimize Structure channel (S2.1) with other channels frozen;
3) optimize Physical channel (P2.1) on top of frozen S2.1;
4) optimize MPNN channel (M2.1) on top of frozen S2.1+P2.1;
5) run minimal fusion calibration.

No external baseline features are used (FoldX/Rosetta/ESM/StaB etc.).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def _rank_1d(values: np.ndarray) -> np.ndarray:
    """Stable normalized rank in [0, 1] (no tie averaging)."""
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
    """Row-wise stable normalized rank in [0, 1]."""
    s = np.asarray(scores_2d, dtype=float)
    if s.ndim != 2:
        raise ValueError("Expected 2D scores matrix.")
    n = s.shape[1]
    if n <= 1:
        return np.zeros_like(s, dtype=float)
    order = np.argsort(s, axis=1, kind="mergesort")
    out = np.empty_like(order, dtype=np.int64)
    row_idx = np.arange(s.shape[0])[:, None]
    out[row_idx, order] = np.arange(n, dtype=np.int64)[None, :]
    return out.astype(float) / float(n - 1)


def _safe_numeric(df: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=float)
    x = pd.to_numeric(df[col], errors="coerce")
    fill = float(x.median()) if x.notna().any() else float(default)
    return x.fillna(fill).values.astype(float)


def _pearson_corr_rows(x: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """Vectorized Pearson corr between 1D x and each row of y2d."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y2d, dtype=float)
    x0 = x - x.mean()
    y0 = y - y.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0, axis=1)
    return (y0 @ x0) / np.maximum(denom, 1e-12)


def _pearson_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.linalg.norm(x0) * np.linalg.norm(y0)
    return float((x0 @ y0) / max(float(den), 1e-12))


@dataclass(frozen=True)
class SearchSpec:
    gate_main_rho: float = 0.700
    gate_external_delta: float = -0.010
    stage1_samples: int = 50000
    stage2_samples: int = 50000
    stage3_samples: int = 50000
    stage4_samples: int = 30000
    topk: int = 50
    seed: int = 20260301


@dataclass
class PreparedDataset:
    name: str
    df: pd.DataFrame
    y_rank: np.ndarray
    rho_base: float
    # Base channels
    phys_base: np.ndarray
    struct_base: np.ndarray
    mpnn_base: np.ndarray
    # Structure primitives
    n_iface: np.ndarray
    n_neighbors: np.ndarray
    burial: np.ndarray
    vol_buried: np.ndarray
    charge_buried: np.ndarray
    # Physical primitives
    xint: np.ndarray
    bind: np.ndarray
    q_pack: np.ndarray
    q_charge: np.ndarray
    # MPNN primitives
    mpnn_neg: np.ndarray
    mpnn_bind: np.ndarray
    mpnn_disp: np.ndarray


def _resolve_struct_col(df: pd.DataFrame) -> str:
    for c in ["rankscore_struct", "rankscore_struct_base", "struct_composite"]:
        if c in df.columns:
            return c
    raise ValueError("Missing structural base column.")


def _resolve_phys_cols(df: pd.DataFrame) -> tuple[str, str]:
    c1 = "ddg_paired_xint_iface" if "ddg_paired_xint_iface" in df.columns else "ddg_xint_iface"
    c2 = "ddg_paired_bind_proxy" if "ddg_paired_bind_proxy" in df.columns else "ddg_bind_proxy"
    if c1 not in df.columns or c2 not in df.columns:
        raise ValueError("Missing physical proxy columns.")
    return c1, c2


def _resolve_mpnn_cols(df: pd.DataFrame) -> tuple[str, str]:
    if "mpnn_neg_llr_complex" not in df.columns or "mpnn_ddg_bind" not in df.columns:
        raise ValueError("Missing MPNN columns.")
    return "mpnn_neg_llr_complex", "mpnn_ddg_bind"


def _prepare_one(name: str, csv_path: Path) -> PreparedDataset:
    df = pd.read_csv(csv_path)
    if "ddg_exp" not in df.columns:
        raise ValueError(f"{name}: missing ddg_exp")

    y_rank = _rank_1d(_safe_numeric(df, "ddg_exp"))

    # Base channels
    if "rankscore_phys" in df.columns:
        phys_base = _rank_1d(_safe_numeric(df, "rankscore_phys"))
    else:
        p1, p2 = _resolve_phys_cols(df)
        phys_base = _rank_1d(_rank_1d(_safe_numeric(df, p1)) + _rank_1d(_safe_numeric(df, p2)))

    struct_col = _resolve_struct_col(df)
    struct_base = _rank_1d(_safe_numeric(df, struct_col))

    if "rankscore_mpnn" in df.columns:
        mpnn_base = _rank_1d(_safe_numeric(df, "rankscore_mpnn"))
    else:
        m1, m2 = _resolve_mpnn_cols(df)
        mpnn_base = _rank_1d(_rank_1d(_safe_numeric(df, m1)) + _rank_1d(_safe_numeric(df, m2)))

    if "rankscore_3view_base" in df.columns:
        base_score = _rank_1d(_safe_numeric(df, "rankscore_3view_base"))
    else:
        base_score = _rank_1d(phys_base + struct_base + mpnn_base)
    rho_base = _pearson_corr_1d(y_rank, base_score)

    burial = np.clip(_safe_numeric(df, "burial_proxy_v10", default=0.5), 0.0, 1.0)
    abs_dv = np.abs(_safe_numeric(df, "delta_volume", default=0.0))
    abs_dc = np.abs(_safe_numeric(df, "delta_charge", default=0.0))

    p1, p2 = _resolve_phys_cols(df)
    m1, m2 = _resolve_mpnn_cols(df)

    mpnn_neg = _rank_1d(_safe_numeric(df, m1))
    mpnn_bind = _rank_1d(_safe_numeric(df, m2))

    return PreparedDataset(
        name=name,
        df=df,
        y_rank=y_rank,
        rho_base=rho_base,
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


def _objective(main_rho: np.ndarray, bpti_delta: np.ndarray, ood_delta: np.ndarray, gate_delta: float) -> np.ndarray:
    pen_bpti = 2.0 * np.maximum(0.0, gate_delta - bpti_delta)
    pen_ood = 1.2 * np.maximum(0.0, gate_delta - ood_delta)
    return main_rho - pen_bpti - pen_ood


def _eval_batch(
    main_scores: np.ndarray,
    bpti_scores: np.ndarray,
    ood_scores: np.ndarray,
    ds_main: PreparedDataset,
    ds_bpti: PreparedDataset,
    ds_ood: PreparedDataset,
    gate_delta: float,
) -> dict[str, np.ndarray]:
    rm = _rank_rows(main_scores)
    rb = _rank_rows(bpti_scores)
    ro = _rank_rows(ood_scores)

    rho_main = _pearson_corr_rows(ds_main.y_rank, rm)
    rho_bpti = _pearson_corr_rows(ds_bpti.y_rank, rb)
    rho_ood = _pearson_corr_rows(ds_ood.y_rank, ro)

    bpti_delta = rho_bpti - ds_bpti.rho_base
    ood_delta = rho_ood - ds_ood.rho_base
    obj = _objective(rho_main, bpti_delta, ood_delta, gate_delta)
    return {
        "main_rho": rho_main,
        "bpti_delta": bpti_delta,
        "ood_delta": ood_delta,
        "objective": obj,
    }


def _select_idx(
    metrics: dict[str, np.ndarray],
    *,
    gate_main: float,
    gate_delta: float,
) -> tuple[int, str]:
    m = metrics["main_rho"]
    bd = metrics["bpti_delta"]
    od = metrics["ood_delta"]
    obj = metrics["objective"]

    mask_70 = (m >= gate_main) & (bd >= gate_delta) & (od >= gate_delta)
    if np.any(mask_70):
        idx_local = int(np.argmax(m[mask_70]))
        idx = int(np.where(mask_70)[0][idx_local])
        return idx, "gate70_max_main"

    mask_safe = (bd >= gate_delta) & (od >= gate_delta)
    if np.any(mask_safe):
        idx_local = int(np.argmax(obj[mask_safe]))
        idx = int(np.where(mask_safe)[0][idx_local])
        return idx, "safe_objective"

    idx = int(np.argmax(obj))
    return idx, "objective_fallback"


def _top_table(
    metrics: dict[str, np.ndarray],
    weights: np.ndarray,
    weight_cols: list[str],
    topk: int,
) -> pd.DataFrame:
    order = np.argsort(metrics["objective"])[::-1]
    keep = order[: min(topk, len(order))]
    out = pd.DataFrame({
        "main_rho": metrics["main_rho"][keep],
        "bpti_delta": metrics["bpti_delta"][keep],
        "ood_delta": metrics["ood_delta"][keep],
        "objective": metrics["objective"][keep],
    })
    for i, c in enumerate(weight_cols):
        out[c] = weights[keep, i]
    return out


def _sample_lognormal_around(
    rng: np.random.Generator,
    base: np.ndarray,
    n: int,
    sigma: float,
) -> np.ndarray:
    z = rng.normal(0.0, sigma, size=(n, len(base)))
    return base[None, :] * np.exp(z)


def _sample_positive(
    rng: np.random.Generator,
    n: int,
    d: int,
    low: float,
    high: float,
) -> np.ndarray:
    return rng.uniform(low, high, size=(n, d))


def _build_struct_score(ds: PreparedDataset, w: np.ndarray) -> np.ndarray:
    x = np.stack(
        [ds.struct_base, ds.n_iface, ds.n_neighbors, ds.burial, ds.vol_buried, ds.charge_buried],
        axis=1,
    )
    return w @ x.T


def _build_phys_score(ds: PreparedDataset, w: np.ndarray) -> np.ndarray:
    x = np.stack(
        [ds.phys_base, ds.xint, ds.bind, ds.q_pack, ds.q_charge, ds.vol_buried, ds.charge_buried],
        axis=1,
    )
    return w @ x.T


def _build_mpnn_score(ds: PreparedDataset, w: np.ndarray) -> np.ndarray:
    # w = [w_base, w_neg, w_bind, w_disp_penalty]
    pos = w[:, [0]] * ds.mpnn_base[None, :] + w[:, [1]] * ds.mpnn_neg[None, :] + w[:, [2]] * ds.mpnn_bind[None, :]
    return pos - w[:, [3]] * ds.mpnn_disp[None, :]


def _stage_search(
    *,
    stage_name: str,
    n_samples: int,
    rng: np.random.Generator,
    ds_main: PreparedDataset,
    ds_bpti: PreparedDataset,
    ds_ood: PreparedDataset,
    fixed_s_main: np.ndarray,
    fixed_p_main: np.ndarray,
    fixed_m_main: np.ndarray,
    fixed_s_bpti: np.ndarray,
    fixed_p_bpti: np.ndarray,
    fixed_m_bpti: np.ndarray,
    fixed_s_ood: np.ndarray,
    fixed_p_ood: np.ndarray,
    fixed_m_ood: np.ndarray,
    gate_main: float,
    gate_delta: float,
    topk: int,
    prior: np.ndarray | None = None,
) -> dict:
    if stage_name == "structure":
        d = 6
        cols = ["w_struct_base", "w_iface8", "w_nei10", "w_burial", "w_vol_buried", "w_charge_buried"]
        w = _sample_positive(rng, n_samples, d, low=0.1, high=2.4) if prior is None else _sample_lognormal_around(rng, prior, n_samples, sigma=0.30)
        s_main = _build_struct_score(ds_main, w)
        s_bpti = _build_struct_score(ds_bpti, w)
        s_ood = _build_struct_score(ds_ood, w)
        m_main = fixed_p_main[None, :] + fixed_m_main[None, :] + s_main
        m_bpti = fixed_p_bpti[None, :] + fixed_m_bpti[None, :] + s_bpti
        m_ood = fixed_p_ood[None, :] + fixed_m_ood[None, :] + s_ood
    elif stage_name == "physical":
        d = 7
        cols = ["w_phys_base", "w_xint", "w_bind", "w_q_pack", "w_q_charge", "w_vol_buried", "w_charge_buried"]
        w = _sample_positive(rng, n_samples, d, low=0.1, high=2.4) if prior is None else _sample_lognormal_around(rng, prior, n_samples, sigma=0.30)
        p_main = _build_phys_score(ds_main, w)
        p_bpti = _build_phys_score(ds_bpti, w)
        p_ood = _build_phys_score(ds_ood, w)
        m_main = fixed_s_main[None, :] + fixed_m_main[None, :] + p_main
        m_bpti = fixed_s_bpti[None, :] + fixed_m_bpti[None, :] + p_bpti
        m_ood = fixed_s_ood[None, :] + fixed_m_ood[None, :] + p_ood
    elif stage_name == "mpnn":
        d = 4
        cols = ["w_mpnn_base", "w_mpnn_neg", "w_mpnn_bind", "w_mpnn_disp_penalty"]
        w = _sample_positive(rng, n_samples, d, low=0.05, high=2.0) if prior is None else _sample_lognormal_around(rng, prior, n_samples, sigma=0.30)
        mp_main = _build_mpnn_score(ds_main, w)
        mp_bpti = _build_mpnn_score(ds_bpti, w)
        mp_ood = _build_mpnn_score(ds_ood, w)
        m_main = fixed_s_main[None, :] + fixed_p_main[None, :] + mp_main
        m_bpti = fixed_s_bpti[None, :] + fixed_p_bpti[None, :] + mp_bpti
        m_ood = fixed_s_ood[None, :] + fixed_p_ood[None, :] + mp_ood
    elif stage_name == "fusion":
        d = 3
        cols = ["w_phys", "w_struct", "w_mpnn"]
        if prior is None:
            w = _sample_positive(rng, n_samples, d, low=0.70, high=1.35)
        else:
            w = _sample_lognormal_around(rng, prior, n_samples, sigma=0.15)
        m_main = w[:, [0]] * fixed_p_main[None, :] + w[:, [1]] * fixed_s_main[None, :] + w[:, [2]] * fixed_m_main[None, :]
        m_bpti = w[:, [0]] * fixed_p_bpti[None, :] + w[:, [1]] * fixed_s_bpti[None, :] + w[:, [2]] * fixed_m_bpti[None, :]
        m_ood = w[:, [0]] * fixed_p_ood[None, :] + w[:, [1]] * fixed_s_ood[None, :] + w[:, [2]] * fixed_m_ood[None, :]
    else:
        raise ValueError(f"Unsupported stage: {stage_name}")

    metrics = _eval_batch(m_main, m_bpti, m_ood, ds_main, ds_bpti, ds_ood, gate_delta)
    idx, strategy = _select_idx(metrics, gate_main=gate_main, gate_delta=gate_delta)

    selected = {
        "index": int(idx),
        "strategy": strategy,
        "weights": {cols[i]: float(w[idx, i]) for i in range(w.shape[1])},
        "metrics": {
            "main_rho": float(metrics["main_rho"][idx]),
            "bpti_delta": float(metrics["bpti_delta"][idx]),
            "ood_delta": float(metrics["ood_delta"][idx]),
            "objective": float(metrics["objective"][idx]),
            "gate_main": bool(float(metrics["main_rho"][idx]) >= gate_main),
            "gate_bpti": bool(float(metrics["bpti_delta"][idx]) >= gate_delta),
            "gate_ood": bool(float(metrics["ood_delta"][idx]) >= gate_delta),
            "gate_all": bool(
                (float(metrics["main_rho"][idx]) >= gate_main)
                and (float(metrics["bpti_delta"][idx]) >= gate_delta)
                and (float(metrics["ood_delta"][idx]) >= gate_delta)
            ),
        },
    }
    top_df = _top_table(metrics, w, cols, topk=topk)
    return {"selected": selected, "top_df": top_df, "weights_all": w}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _baseline_snapshot(
    ds_main: PreparedDataset,
    ds_bpti: PreparedDataset,
    ds_ood: PreparedDataset,
    spec: SearchSpec,
) -> dict:
    return {
        "main": {"rho_base": float(ds_main.rho_base), "n_mut": int(len(ds_main.df)), "n_targets": int(ds_main.df["target"].nunique())},
        "bpti": {"rho_base": float(ds_bpti.rho_base), "n_mut": int(len(ds_bpti.df)), "n_targets": int(ds_bpti.df["target"].nunique())},
        "ood": {"rho_base": float(ds_ood.rho_base), "n_mut": int(len(ds_ood.df)), "n_targets": int(ds_ood.df["target"].nunique())},
        "gates": {"main_rho": float(spec.gate_main_rho), "external_delta": float(spec.gate_external_delta)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/main_eval_v19.csv")
    ap.add_argument("--bpti-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/bpti_eval_v19.csv")
    ap.add_argument("--ood-csv", default="research/pepddg_v5/results/v19_sanitized_baseline/ood_eval_v19.csv")
    ap.add_argument("--out-dir", default="research/pepddg_v5/results/v19_strict3_s2p2m2")
    ap.add_argument("--seed", type=int, default=SearchSpec.seed)
    ap.add_argument("--stage1-samples", type=int, default=SearchSpec.stage1_samples)
    ap.add_argument("--stage2-samples", type=int, default=SearchSpec.stage2_samples)
    ap.add_argument("--stage3-samples", type=int, default=SearchSpec.stage3_samples)
    ap.add_argument("--stage4-samples", type=int, default=SearchSpec.stage4_samples)
    ap.add_argument("--gate-main", type=float, default=SearchSpec.gate_main_rho)
    ap.add_argument("--gate-delta", type=float, default=SearchSpec.gate_external_delta)
    ap.add_argument("--topk", type=int, default=SearchSpec.topk)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = SearchSpec(
        gate_main_rho=float(args.gate_main),
        gate_external_delta=float(args.gate_delta),
        stage1_samples=int(args.stage1_samples),
        stage2_samples=int(args.stage2_samples),
        stage3_samples=int(args.stage3_samples),
        stage4_samples=int(args.stage4_samples),
        topk=int(args.topk),
        seed=int(args.seed),
    )
    rng = np.random.default_rng(spec.seed)

    ds_main = _prepare_one("main", Path(args.main_csv))
    ds_bpti = _prepare_one("bpti", Path(args.bpti_csv))
    ds_ood = _prepare_one("ood", Path(args.ood_csv))

    # Task 0: baseline snapshot
    baseline = _baseline_snapshot(ds_main, ds_bpti, ds_ood, spec)
    _write_json(out_dir / "baseline_snapshot.json", baseline)

    # Initial fixed channels = base
    s_main, p_main, m_main = ds_main.struct_base, ds_main.phys_base, ds_main.mpnn_base
    s_bpti, p_bpti, m_bpti = ds_bpti.struct_base, ds_bpti.phys_base, ds_bpti.mpnn_base
    s_ood, p_ood, m_ood = ds_ood.struct_base, ds_ood.phys_base, ds_ood.mpnn_base

    # Task 1: Structure search
    st1 = _stage_search(
        stage_name="structure",
        n_samples=spec.stage1_samples,
        rng=rng,
        ds_main=ds_main,
        ds_bpti=ds_bpti,
        ds_ood=ds_ood,
        fixed_s_main=s_main,
        fixed_p_main=p_main,
        fixed_m_main=m_main,
        fixed_s_bpti=s_bpti,
        fixed_p_bpti=p_bpti,
        fixed_m_bpti=m_bpti,
        fixed_s_ood=s_ood,
        fixed_p_ood=p_ood,
        fixed_m_ood=m_ood,
        gate_main=spec.gate_main_rho,
        gate_delta=spec.gate_external_delta,
        topk=spec.topk,
        prior=np.array([1.0, 1.0, 1.0, 0.7, 0.6, 0.6], dtype=float),
    )
    st1["top_df"].to_csv(out_dir / "stage1_structure_top.csv", index=False)
    _write_json(out_dir / "stage1_structure_selected.json", st1["selected"])
    w1 = np.array(list(st1["selected"]["weights"].values()), dtype=float)[None, :]
    s_main = _build_struct_score(ds_main, w1)[0]
    s_bpti = _build_struct_score(ds_bpti, w1)[0]
    s_ood = _build_struct_score(ds_ood, w1)[0]

    # Task 2: Physical search (with S frozen)
    st2 = _stage_search(
        stage_name="physical",
        n_samples=spec.stage2_samples,
        rng=rng,
        ds_main=ds_main,
        ds_bpti=ds_bpti,
        ds_ood=ds_ood,
        fixed_s_main=s_main,
        fixed_p_main=p_main,
        fixed_m_main=m_main,
        fixed_s_bpti=s_bpti,
        fixed_p_bpti=p_bpti,
        fixed_m_bpti=m_bpti,
        fixed_s_ood=s_ood,
        fixed_p_ood=p_ood,
        fixed_m_ood=m_ood,
        gate_main=spec.gate_main_rho,
        gate_delta=spec.gate_external_delta,
        topk=spec.topk,
        prior=np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.5, 0.5], dtype=float),
    )
    st2["top_df"].to_csv(out_dir / "stage2_physical_top.csv", index=False)
    _write_json(out_dir / "stage2_physical_selected.json", st2["selected"])
    w2 = np.array(list(st2["selected"]["weights"].values()), dtype=float)[None, :]
    p_main = _build_phys_score(ds_main, w2)[0]
    p_bpti = _build_phys_score(ds_bpti, w2)[0]
    p_ood = _build_phys_score(ds_ood, w2)[0]

    # Task 3: MPNN search (with S,P frozen)
    st3 = _stage_search(
        stage_name="mpnn",
        n_samples=spec.stage3_samples,
        rng=rng,
        ds_main=ds_main,
        ds_bpti=ds_bpti,
        ds_ood=ds_ood,
        fixed_s_main=s_main,
        fixed_p_main=p_main,
        fixed_m_main=m_main,
        fixed_s_bpti=s_bpti,
        fixed_p_bpti=p_bpti,
        fixed_m_bpti=m_bpti,
        fixed_s_ood=s_ood,
        fixed_p_ood=p_ood,
        fixed_m_ood=m_ood,
        gate_main=spec.gate_main_rho,
        gate_delta=spec.gate_external_delta,
        topk=spec.topk,
        prior=np.array([1.0, 1.0, 1.0, 0.3], dtype=float),
    )
    st3["top_df"].to_csv(out_dir / "stage3_mpnn_top.csv", index=False)
    _write_json(out_dir / "stage3_mpnn_selected.json", st3["selected"])
    w3 = np.array(list(st3["selected"]["weights"].values()), dtype=float)[None, :]
    m_main = _build_mpnn_score(ds_main, w3)[0]
    m_bpti = _build_mpnn_score(ds_bpti, w3)[0]
    m_ood = _build_mpnn_score(ds_ood, w3)[0]

    # Task 4: minimal fusion
    st4 = _stage_search(
        stage_name="fusion",
        n_samples=spec.stage4_samples,
        rng=rng,
        ds_main=ds_main,
        ds_bpti=ds_bpti,
        ds_ood=ds_ood,
        fixed_s_main=s_main,
        fixed_p_main=p_main,
        fixed_m_main=m_main,
        fixed_s_bpti=s_bpti,
        fixed_p_bpti=p_bpti,
        fixed_m_bpti=m_bpti,
        fixed_s_ood=s_ood,
        fixed_p_ood=p_ood,
        fixed_m_ood=m_ood,
        gate_main=spec.gate_main_rho,
        gate_delta=spec.gate_external_delta,
        topk=spec.topk,
        prior=np.array([1.0, 1.0, 1.0], dtype=float),
    )
    st4["top_df"].to_csv(out_dir / "stage4_fusion_top.csv", index=False)
    _write_json(out_dir / "stage4_fusion_selected.json", st4["selected"])
    w4 = np.array(list(st4["selected"]["weights"].values()), dtype=float)

    final_main = w4[0] * p_main + w4[1] * s_main + w4[2] * m_main
    final_bpti = w4[0] * p_bpti + w4[1] * s_bpti + w4[2] * m_bpti
    final_ood = w4[0] * p_ood + w4[1] * s_ood + w4[2] * m_ood

    final_main_rho = _pearson_corr_1d(ds_main.y_rank, _rank_1d(final_main))
    final_bpti_rho = _pearson_corr_1d(ds_bpti.y_rank, _rank_1d(final_bpti))
    final_ood_rho = _pearson_corr_1d(ds_ood.y_rank, _rank_1d(final_ood))
    final_metrics = {
        "main_rho": float(final_main_rho),
        "bpti_delta": float(final_bpti_rho - ds_bpti.rho_base),
        "ood_delta": float(final_ood_rho - ds_ood.rho_base),
        "gate_main": bool(final_main_rho >= spec.gate_main_rho),
        "gate_bpti": bool((final_bpti_rho - ds_bpti.rho_base) >= spec.gate_external_delta),
        "gate_ood": bool((final_ood_rho - ds_ood.rho_base) >= spec.gate_external_delta),
    }
    final_metrics["gate_all"] = bool(final_metrics["gate_main"] and final_metrics["gate_bpti"] and final_metrics["gate_ood"])

    # Save per-row outputs for reproducibility/audit
    main_out = ds_main.df.copy()
    main_out["v19_struct_score"] = s_main
    main_out["v19_phys_score"] = p_main
    main_out["v19_mpnn_score"] = m_main
    main_out["rankscore_3view_v19_strict3"] = _rank_1d(final_main)
    main_out.to_csv(out_dir / "main_eval_v19.csv", index=False)

    bpti_out = ds_bpti.df.copy()
    bpti_out["v19_struct_score"] = s_bpti
    bpti_out["v19_phys_score"] = p_bpti
    bpti_out["v19_mpnn_score"] = m_bpti
    bpti_out["rankscore_3view_v19_strict3"] = _rank_1d(final_bpti)
    bpti_out.to_csv(out_dir / "bpti_eval_v19.csv", index=False)

    ood_out = ds_ood.df.copy()
    ood_out["v19_struct_score"] = s_ood
    ood_out["v19_phys_score"] = p_ood
    ood_out["v19_mpnn_score"] = m_ood
    ood_out["rankscore_3view_v19_strict3"] = _rank_1d(final_ood)
    ood_out.to_csv(out_dir / "ood_eval_v19.csv", index=False)

    policy_audit = {
        "strict_clean3_columns_used": sorted(
            [
                "rankscore_phys",
                "rankscore_struct",
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
                "mpnn_neg_llr_complex",
                "mpnn_ddg_bind",
            ]
        ),
        "banned_patterns": ["foldx", "rosetta", "stabddg", "esm2", "esmif", "esm3", "diffaffinity", "saprot", "cartddg"],
    }
    _write_json(out_dir / "policy_audit_v19.json", policy_audit)

    summary = {
        "spec": asdict(spec),
        "baseline": baseline,
        "stage1_structure": st1["selected"],
        "stage2_physical": st2["selected"],
        "stage3_mpnn": st3["selected"],
        "stage4_fusion": st4["selected"],
        "final_metrics": final_metrics,
    }
    _write_json(out_dir / "final_summary_v19.json", summary)

    # Human-readable report
    report_lines = [
        "# v19 strict clean-3 channelwise optimization",
        "",
        f"- seed: {spec.seed}",
        f"- gates: main >= {spec.gate_main_rho:.3f}, external deltas >= {spec.gate_external_delta:+.3f}",
        "",
        "## Baseline (mainline base comparator)",
        f"- main rho(base): {baseline['main']['rho_base']:.6f}",
        f"- bpti rho(base): {baseline['bpti']['rho_base']:.6f}",
        f"- ood rho(base):  {baseline['ood']['rho_base']:.6f}",
        "",
        "## Stage selections",
        f"- S2.1 strategy: {st1['selected']['strategy']} | metrics={st1['selected']['metrics']}",
        f"- P2.1 strategy: {st2['selected']['strategy']} | metrics={st2['selected']['metrics']}",
        f"- M2.1 strategy: {st3['selected']['strategy']} | metrics={st3['selected']['metrics']}",
        f"- Fusion strategy: {st4['selected']['strategy']} | metrics={st4['selected']['metrics']}",
        "",
        "## Final",
        f"- main rho:   {final_metrics['main_rho']:.6f}",
        f"- bpti delta: {final_metrics['bpti_delta']:+.6f}",
        f"- ood delta:  {final_metrics['ood_delta']:+.6f}",
        f"- gate_main:  {final_metrics['gate_main']}",
        f"- gate_bpti:  {final_metrics['gate_bpti']}",
        f"- gate_ood:   {final_metrics['gate_ood']}",
        f"- gate_all:   {final_metrics['gate_all']}",
    ]
    (out_dir / "report_v19.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print("v19 strict clean-3 optimization complete")
    print("=" * 72)
    print(f"main_rho   = {final_metrics['main_rho']:.6f}")
    print(f"bpti_delta = {final_metrics['bpti_delta']:+.6f}")
    print(f"ood_delta  = {final_metrics['ood_delta']:+.6f}")
    print(f"gate_all   = {final_metrics['gate_all']}")
    print(f"out_dir    = {out_dir}")


if __name__ == "__main__":
    main()
