#!/usr/bin/env python
"""Common helpers for v12 follow-up: atlas, trust routing, and clipping."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PRED_COLS = {
    "base": "rankscore_3view_base",
    "target065": "rankscore_3view_target065",
    "balanced": "rankscore_3view_balanced",
    "mainmax": "rankscore_3view_p3_v12_p123_mainmax",
    "tradeoff": "rankscore_3view_p3_v12_p123_tradeoff",
    "safe": "rankscore_3view_p3_v12_p123_safefallback",
}


@dataclass(frozen=True)
class TrustRouteConfig:
    """Deterministic trust-score routing config."""

    name: str
    w_rel: float
    w_disp: float
    w_risk: float
    w_local: float
    w_burial_mid: float
    pen_charge_buried: float
    pen_large_volume: float
    t_high: float
    t_mid: float
    disp_guard: float
    force_safe_on_combo: bool


@dataclass(frozen=True)
class ClipConfig:
    """Deterministic regime-specific clipping config."""

    name: str
    risk_q_thr: float
    disp_q_thr: float
    blend_base: float
    blend_charge_buried: float
    blend_large_volume: float
    blend_high_disp_risk: float
    only_non_safe_routes: bool = True


def safe_spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    rho, _ = spearmanr(y, yhat)
    return float(rho) if not np.isnan(rho) else np.nan


def ensure_required_columns(df: pd.DataFrame) -> None:
    req = [
        "ddg_exp",
        "delta_charge",
        "delta_volume",
        "burial_proxy_v10",
        "q_view_dispersion",
        "q_charge",
        "q_pack",
        "local_apply_flag_target065",
        *PRED_COLS.values(),
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic risk/regime descriptors used by follow-up logic."""
    out = df.copy()
    ensure_required_columns(out)

    out["abs_charge"] = out["delta_charge"].abs().astype(float)
    out["abs_volume"] = out["delta_volume"].abs().astype(float)
    out["q_risk"] = np.maximum(out["q_charge"].astype(float).values, out["q_pack"].astype(float).values)

    out["risk_charge_buried"] = (
        (out["abs_charge"] >= 1.0) & (out["burial_proxy_v10"].astype(float) >= 0.60)
    ).astype(float)
    out["risk_large_volume"] = (out["abs_volume"] >= 85.0).astype(float)
    out["risk_combo"] = ((out["risk_charge_buried"] > 0.0) | (out["risk_large_volume"] > 0.0)).astype(float)

    out["burial_midness"] = 1.0 - (out["burial_proxy_v10"].astype(float) - 0.5).abs() / 0.5
    out["burial_midness"] = out["burial_midness"].clip(0.0, 1.0)

    out["rel_base"] = 1.0 - 0.55 * out["q_view_dispersion"].astype(float) - 0.45 * out["q_risk"].astype(float)
    out["rel_base"] = out["rel_base"].clip(0.0, 1.0)

    out["regime_tag"] = np.where(
        out["risk_charge_buried"] > 0.0,
        "charge_buried",
        np.where(out["risk_large_volume"] > 0.0, "large_volume", "other"),
    )
    return out


def trust_score(df: pd.DataFrame, cfg: TrustRouteConfig) -> np.ndarray:
    """Compute normalized trust score in [0, 1]."""
    rel = df["rel_base"].astype(float).values
    inv_disp = 1.0 - df["q_view_dispersion"].astype(float).values
    inv_risk = 1.0 - df["q_risk"].astype(float).values
    local = df["local_apply_flag_target065"].astype(float).values
    burial_mid = df["burial_midness"].astype(float).values

    charge_pen = df["risk_charge_buried"].astype(float).values
    volume_pen = df["risk_large_volume"].astype(float).values

    positive = (
        cfg.w_rel * rel
        + cfg.w_disp * inv_disp
        + cfg.w_risk * inv_risk
        + cfg.w_local * local
        + cfg.w_burial_mid * burial_mid
    )
    negative = cfg.pen_charge_buried * charge_pen + cfg.pen_large_volume * volume_pen
    raw = positive - negative

    raw_min = -cfg.pen_charge_buried - cfg.pen_large_volume
    raw_max = cfg.w_rel + cfg.w_disp + cfg.w_risk + cfg.w_local + cfg.w_burial_mid
    score = (raw - raw_min) / (raw_max - raw_min + 1e-12)
    return np.clip(score, 0.0, 1.0)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method="average", pct=True).values


def route_prediction(df: pd.DataFrame, cfg: TrustRouteConfig) -> pd.DataFrame:
    """Route each sample to mainmax/tradeoff/safe using deterministic trust rules."""
    out = df.copy()
    t = trust_score(out, cfg)

    risk_combo = out["risk_combo"].astype(float).values > 0.0
    charge_buried = out["risk_charge_buried"].astype(float).values > 0.0
    large_volume = out["risk_large_volume"].astype(float).values > 0.0
    disp = out["q_view_dispersion"].astype(float).values

    use_mainmax = (t >= cfg.t_high) & (~charge_buried) & (~large_volume)
    use_tradeoff = (t >= cfg.t_mid) & (disp <= cfg.disp_guard)
    use_safe = ~(use_mainmax | use_tradeoff)

    if cfg.force_safe_on_combo:
        use_safe = use_safe | risk_combo
        use_mainmax = use_mainmax & (~risk_combo)
        use_tradeoff = use_tradeoff & (~risk_combo)

    route = np.where(use_mainmax, "mainmax", np.where(use_tradeoff, "tradeoff", "safe"))

    # Route on percentile-calibrated variant scores to make cross-variant
    # sample-level switching comparable in scale.
    pred_mainmax = _percentile_rank(out[PRED_COLS["mainmax"]].astype(float).values)
    pred_tradeoff = _percentile_rank(out[PRED_COLS["tradeoff"]].astype(float).values)
    pred_safe = _percentile_rank(out[PRED_COLS["safe"]].astype(float).values)
    pred = np.where(route == "mainmax", pred_mainmax, np.where(route == "tradeoff", pred_tradeoff, pred_safe))

    out["trust_score"] = t
    out["route_label"] = route
    out["pred_route"] = pred
    return out


def apply_regime_clipping(df: pd.DataFrame, cfg: ClipConfig) -> pd.DataFrame:
    """Apply deterministic clipping from routed prediction toward safe prediction."""
    out = df.copy()
    if "pred_route" not in out.columns or "route_label" not in out.columns:
        raise ValueError("route columns missing; call route_prediction first")

    q_risk = out["q_risk"].astype(float).values
    q_disp = out["q_view_dispersion"].astype(float).values
    charge_buried = out["risk_charge_buried"].astype(float).values
    large_volume = out["risk_large_volume"].astype(float).values

    high_disp_risk = ((q_risk >= cfg.risk_q_thr) & (q_disp >= cfg.disp_q_thr)).astype(float)
    strength = (
        cfg.blend_base
        + cfg.blend_charge_buried * charge_buried
        + cfg.blend_large_volume * large_volume
        + cfg.blend_high_disp_risk * high_disp_risk
    )
    strength = np.clip(strength, 0.0, 1.0)

    if cfg.only_non_safe_routes:
        safe_mask = (out["route_label"].values == "safe").astype(float)
        strength = strength * (1.0 - safe_mask)

    pred_route = out["pred_route"].astype(float).values
    pred_safe = _percentile_rank(out[PRED_COLS["safe"]].astype(float).values)
    pred_clip = (1.0 - strength) * pred_route + strength * pred_safe

    out["clip_strength"] = strength
    out["clip_high_disp_risk"] = high_disp_risk
    out["pred_clipped"] = pred_clip
    out["clip_flag"] = (strength > 1e-12).astype(float)
    return out


def eval_prediction(df: pd.DataFrame, pred_col: str) -> dict:
    y = df["ddg_exp"].values.astype(float)
    base = df[PRED_COLS["base"]].values.astype(float)
    pred = df[pred_col].values.astype(float)
    rho_base = safe_spearman(y, base)
    rho_new = safe_spearman(y, pred)
    return {
        "rho_base": float(rho_base),
        "rho_new": float(rho_new),
        "delta": float(rho_new - rho_base),
    }


def gate_summary(
    main_metrics: dict,
    bpti_metrics: dict,
    ood_metrics: dict,
    *,
    gate_main_rho: float = 0.650,
    gate_external_delta: float = -0.010,
) -> dict:
    main_rho = float(main_metrics["rho_new"])
    bpti_delta = float(bpti_metrics["delta"])
    ood_delta = float(ood_metrics["delta"])

    main_gate = bool(main_rho >= gate_main_rho)
    bpti_gate = bool(bpti_delta >= gate_external_delta)
    ood_gate = bool(ood_delta >= gate_external_delta)

    return {
        "main_rho": main_rho,
        "main_delta": float(main_metrics["delta"]),
        "bpti_delta": bpti_delta,
        "ood_delta": ood_delta,
        "main_gate": main_gate,
        "bpti_gate": bpti_gate,
        "ood_gate": ood_gate,
        "gate_pass_all": bool(main_gate and bpti_gate and ood_gate),
        "gate_main_margin": float(main_rho - gate_main_rho),
        "gate_bpti_margin": float(bpti_delta - gate_external_delta),
        "gate_ood_margin": float(ood_delta - gate_external_delta),
    }


def objective_score(summary: dict) -> float:
    """Gate-aware scalar objective used for ranking candidates."""
    return (
        float(summary["main_rho"])
        + 0.80 * min(0.0, float(summary["gate_bpti_margin"]))
        + 0.60 * min(0.0, float(summary["gate_ood_margin"]))
        + 0.30 * min(0.0, float(summary["gate_main_margin"]))
    )


def cfg_to_dict(cfg: TrustRouteConfig | ClipConfig) -> dict:
    return asdict(cfg)


def pareto_frontier_3d(df: pd.DataFrame, a: str, b: str, c: str) -> pd.DataFrame:
    """Return non-dominated rows maximizing (a, b, c)."""
    if df.empty:
        return df.copy()

    ordered = df.sort_values(a, ascending=False).copy()
    frontier_idx: list[int] = []
    frontier_2d: list[tuple[float, float]] = []

    for idx, row in ordered.iterrows():
        b0 = float(row[b])
        c0 = float(row[c])
        dominated = False
        for bf, cf in frontier_2d:
            if bf >= b0 and cf >= c0:
                dominated = True
                break
        if dominated:
            continue

        new_frontier = []
        for bf, cf in frontier_2d:
            if not (b0 >= bf and c0 >= cf):
                new_frontier.append((bf, cf))
        new_frontier.append((b0, c0))
        frontier_2d = new_frontier
        frontier_idx.append(idx)

    return ordered.loc[frontier_idx].copy()


__all__ = [
    "ClipConfig",
    "PRED_COLS",
    "TrustRouteConfig",
    "add_risk_features",
    "apply_regime_clipping",
    "cfg_to_dict",
    "eval_prediction",
    "gate_summary",
    "objective_score",
    "pareto_frontier_3d",
    "route_prediction",
    "safe_spearman",
    "trust_score",
]
