from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import PepDDGConfig


def stable_rank_1d(values: np.ndarray) -> np.ndarray:
    """Stable normalized rank in [0, 1] using mergesort order."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(v, kind="mergesort")
    out = np.empty(n, dtype=np.float64)
    if n == 1:
        out[order] = 0.0
    else:
        out[order] = np.arange(n, dtype=np.float64) / float(n - 1)
    return out


def metric_rank_1d(values: np.ndarray) -> np.ndarray:
    """Tie-aware normalized rank used for metric reporting."""
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


def _pearson_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.linalg.norm(x0) * np.linalg.norm(y0)
    return float((x0 @ y0) / max(float(den), 1e-12))


def _resolve_phys_cols(df: pd.DataFrame) -> tuple[str, str]:
    c1 = "ddg_paired_xint_iface" if "ddg_paired_xint_iface" in df.columns else "ddg_xint_iface"
    c2 = "ddg_paired_bind_proxy" if "ddg_paired_bind_proxy" in df.columns else "ddg_bind_proxy"
    if c1 not in df.columns or c2 not in df.columns:
        raise ValueError(
            "Cannot derive anchor: missing physical columns. "
            "Need one of ddg_paired_xint_iface/ddg_xint_iface and "
            "ddg_paired_bind_proxy/ddg_bind_proxy."
        )
    return c1, c2


def _resolve_struct_col(df: pd.DataFrame) -> str:
    for c in ["rankscore_struct", "rankscore_struct_base", "struct_composite"]:
        if c in df.columns:
            return c
    raise ValueError(
        "Cannot derive anchor: missing structural column. "
        "Need one of rankscore_struct/rankscore_struct_base/struct_composite."
    )


def _compute_mpnn_base_rank(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if "rankscore_mpnn" in df.columns:
        return stable_rank_1d(_safe_numeric(df, "rankscore_mpnn")), ["rankscore_mpnn"]
    if "mpnn_neg_llr_complex" not in df.columns or "mpnn_ddg_bind" not in df.columns:
        raise ValueError("Missing MPNN columns: mpnn_neg_llr_complex and mpnn_ddg_bind are required.")
    m1 = stable_rank_1d(_safe_numeric(df, "mpnn_neg_llr_complex"))
    m2 = stable_rank_1d(_safe_numeric(df, "mpnn_ddg_bind"))
    return stable_rank_1d(m1 + m2), ["mpnn_neg_llr_complex", "mpnn_ddg_bind"]


def _derive_clean3_anchor(df: pd.DataFrame) -> tuple[np.ndarray, str, list[str]]:
    if "rankscore_3view_base" in df.columns:
        return stable_rank_1d(_safe_numeric(df, "rankscore_3view_base")), "rankscore_3view_base", [
            "rankscore_3view_base"
        ]

    used: list[str] = []
    if "rankscore_phys" in df.columns:
        phys = stable_rank_1d(_safe_numeric(df, "rankscore_phys"))
        used.append("rankscore_phys")
    else:
        p1, p2 = _resolve_phys_cols(df)
        phys = stable_rank_1d(stable_rank_1d(_safe_numeric(df, p1)) + stable_rank_1d(_safe_numeric(df, p2)))
        used.extend([p1, p2])

    struct_col = _resolve_struct_col(df)
    struct = stable_rank_1d(_safe_numeric(df, struct_col))
    used.append(struct_col)
    mpnn, mpnn_used = _compute_mpnn_base_rank(df)
    used.extend(mpnn_used)
    return stable_rank_1d(phys + struct + mpnn), "derived_clean3_base", sorted(set(used))


def _resolve_anchor(df: pd.DataFrame, cfg: PepDDGConfig) -> tuple[np.ndarray, str, list[str]]:
    anchor_column = cfg.effective_anchor_column()
    anchor_preference = cfg.effective_anchor_preference()
    if anchor_column and anchor_column in df.columns:
        return stable_rank_1d(_safe_numeric(df, anchor_column)), anchor_column, [anchor_column]

    for col in anchor_preference:
        if col in df.columns:
            return stable_rank_1d(_safe_numeric(df, col)), col, [col]

    if cfg.strict_anchor:
        raise ValueError(
            "No anchor column found. "
            f"anchor_column={anchor_column}, anchor_preference={anchor_preference}"
        )
    return _derive_clean3_anchor(df)


def find_banned_columns(columns: list[str], patterns: list[str]) -> list[str]:
    out: list[str] = []
    for c in columns:
        cl = c.lower()
        if any(re.search(p, cl) for p in patterns):
            out.append(c)
    return sorted(set(out))


def score_dataframe(df: pd.DataFrame, cfg: PepDDGConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply PepDDG scoring on a feature table."""
    cfg.validate()
    mode = cfg.mode.strip().lower()
    if mode == "cal" and ("mpnn_neg_llr_complex" not in df.columns or "mpnn_ddg_bind" not in df.columns):
        raise ValueError("Input is missing required columns for Cal mode: mpnn_neg_llr_complex, mpnn_ddg_bind")
    anchor_rank, anchor_source, anchor_used_cols = _resolve_anchor(df, cfg)
    mpnn_base_used: list[str] = []
    if mode == "cal" or anchor_source == "derived_clean3_base":
        mpnn_base, mpnn_base_used = _compute_mpnn_base_rank(df)
    else:
        mpnn_base = np.zeros(len(df), dtype=float)
    raw = anchor_rank
    score = stable_rank_1d(raw)
    mpnn_neg = np.asarray([], dtype=float)
    mpnn_bind = np.asarray([], dtype=float)
    mpnn_disp = np.asarray([], dtype=float)
    delta_mpnn = np.asarray([], dtype=float)

    if mode == "cal":
        mpnn_neg = stable_rank_1d(_safe_numeric(df, "mpnn_neg_llr_complex"))
        mpnn_bind = stable_rank_1d(_safe_numeric(df, "mpnn_ddg_bind"))
        mpnn_disp = stable_rank_1d(np.abs(mpnn_neg - mpnn_bind))

        w = cfg.weights
        delta_mpnn = (
            w.w_neg * (mpnn_neg - mpnn_base)
            + w.w_bind * (mpnn_bind - mpnn_base)
            - abs(float(w.w_disp_pen)) * mpnn_disp
        )
        raw = anchor_rank + float(w.w_m) * delta_mpnn
        score = stable_rank_1d(raw)

    out = df.copy()
    out[cfg.score_column] = score
    if cfg.write_auxiliary_columns:
        out["pepddg_mode"] = mode
        out["pepddg_anchor_source"] = anchor_source
        out["pepddg_anchor_rank"] = anchor_rank
        if mpnn_base_used:
            out["pepddg_mpnn_base_rank"] = mpnn_base
        if mode == "cal":
            out["pepddg_mpnn_neg_rank"] = mpnn_neg
            out["pepddg_mpnn_bind_rank"] = mpnn_bind
            out["pepddg_mpnn_disp_rank"] = mpnn_disp
            out["pepddg_v19_delta_mpnn"] = delta_mpnn
            out["pepddg_v19_raw"] = raw

    gate_metrics: dict[str, Any] = {
        "available": False,
        "reason": "ddg_exp not found",
    }
    if "ddg_exp" in df.columns:
        y_rank = metric_rank_1d(_safe_numeric(df, "ddg_exp"))
        pred_rank = metric_rank_1d(raw)
        rho_pred = _pearson_corr_1d(y_rank, pred_rank)
        gate_metrics = {
            "available": True,
            "mode": mode,
            "rho_pred": float(rho_pred),
            "baseline_column": None,
            "rho_baseline": None,
            "delta_vs_baseline": None,
        }
        if cfg.baseline_column_for_delta and cfg.baseline_column_for_delta in df.columns:
            baseline_rank = metric_rank_1d(_safe_numeric(df, cfg.baseline_column_for_delta))
            rho_baseline = _pearson_corr_1d(y_rank, baseline_rank)
            gate_metrics["baseline_column"] = cfg.baseline_column_for_delta
            gate_metrics["rho_baseline"] = float(rho_baseline)
            gate_metrics["delta_vs_baseline"] = float(rho_pred - rho_baseline)

    residual_used = ["mpnn_neg_llr_complex", "mpnn_ddg_bind"] if mode == "cal" else []
    used_columns = sorted(set(anchor_used_cols + mpnn_base_used + residual_used))

    details: dict[str, Any] = {
        "mode": mode,
        "score_column": cfg.score_column,
        "anchor_source": anchor_source,
        "anchor_used_columns": sorted(set(anchor_used_cols)),
        "used_columns": used_columns,
        "gate_metrics": gate_metrics,
    }
    return out, details
