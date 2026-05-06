"""MPNN feature extraction for PepDDG upstream pipeline.

Computes log-likelihood ratio features from ProteinMPNN scores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def compute_mpnn_ddg_features(
    complex_logp_wt: float,
    complex_logp_mut: float,
    chain_logp_wt: float,
    chain_logp_mut: float,
) -> dict[str, float]:
    """Compute MPNN DDG features from log-probabilities.

    Formulas:
        LLR_complex = logP(mut_complex) - logP(wt_complex)
        neg_llr_complex = -LLR_complex
        LLR_chain = logP(mut_chain) - logP(wt_chain)
        ddg_bind = -LLR_complex + LLR_chain

    Args:
        complex_logp_wt: Log-probability of WT sequence in complex context.
        complex_logp_mut: Log-probability of mutant sequence in complex context.
        chain_logp_wt: Log-probability of WT sequence in isolated chain context.
        chain_logp_mut: Log-probability of mutant sequence in isolated chain context.

    Returns:
        Dict with mpnn_neg_llr_complex and mpnn_ddg_bind.
    """
    llr_complex = complex_logp_mut - complex_logp_wt
    llr_chain = chain_logp_mut - chain_logp_wt
    neg_llr_complex = -llr_complex
    ddg_bind = -llr_complex + llr_chain

    return {
        "mpnn_neg_llr_complex": float(neg_llr_complex),
        "mpnn_ddg_bind": float(ddg_bind),
    }


def extract_mpnn_from_csv(mpnn_csv: str | Path) -> pd.DataFrame:
    """Parse MPNN results CSV into a feature DataFrame.

    Handles two formats:
    1. Pre-computed: CSV already has mpnn_neg_llr_complex and mpnn_ddg_bind.
    2. Raw log-probs: CSV has mpnn_logp_{wt,mut}_{complex,chain} columns.

    Args:
        mpnn_csv: Path to MPNN results CSV.

    Returns:
        DataFrame with columns: target, mutation, mpnn_neg_llr_complex, mpnn_ddg_bind.
    """
    df = pd.read_csv(mpnn_csv)

    # If already computed, pass through
    if "mpnn_neg_llr_complex" in df.columns and "mpnn_ddg_bind" in df.columns:
        cols = ["target", "mutation", "mpnn_neg_llr_complex", "mpnn_ddg_bind"]
        return df[[c for c in cols if c in df.columns]].copy()

    # Compute from raw log-probs
    required = [
        "mpnn_logp_wt_complex", "mpnn_logp_mut_complex",
        "mpnn_logp_wt_chain", "mpnn_logp_mut_chain",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"MPNN CSV missing columns: {missing}")

    out = df[["target", "mutation"]].copy()
    llr_complex = df["mpnn_logp_mut_complex"] - df["mpnn_logp_wt_complex"]
    llr_chain = df["mpnn_logp_mut_chain"] - df["mpnn_logp_wt_chain"]
    out["mpnn_neg_llr_complex"] = -llr_complex
    out["mpnn_ddg_bind"] = -llr_complex + llr_chain
    return out


def compute_mpnn_rankscore(neg_llr: np.ndarray, ddg_bind: np.ndarray) -> np.ndarray:
    """Compute MPNN rankscore = rank(rank(neg_llr) + rank(ddg_bind)), normalized.

    Uses ordinal ranking for determinism.

    Args:
        neg_llr: Array of mpnn_neg_llr_complex values.
        ddg_bind: Array of mpnn_ddg_bind values.

    Returns:
        Normalized rank scores in [0, 1].
    """
    neg_llr = np.asarray(neg_llr, dtype=float)
    ddg_bind = np.asarray(ddg_bind, dtype=float)
    n = len(neg_llr)
    if n <= 1:
        return np.zeros(n, dtype=float)

    r1 = rankdata(neg_llr, method="ordinal")
    r2 = rankdata(ddg_bind, method="ordinal")
    combined = r1 + r2
    r_final = rankdata(combined, method="ordinal")
    return (r_final - 1.0) / float(n - 1)
