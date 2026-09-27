"""Strict public feature-table interface for cohort-relative PepDDG ranking."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config import PepDDGConfig
from .scoring import score_dataframe
from .structural_features import compute_struct_composite


IDENTITY_COLUMNS = ("target", "parent_id", "mutation")
FEATURE_COLUMNS = (
    "ddg_xint_iface",
    "ddg_bind_proxy",
    "n_iface_contacts_8a",
    "n_neighbors_10a",
    "mpnn_neg_llr_complex",
    "mpnn_ddg_bind",
)


@dataclass(frozen=True)
class ScoreResult:
    """One successfully validated and scored mutation cohort."""

    table: pd.DataFrame
    summary: dict[str, Any]


def _validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("features must contain at least one mutation")
    if frame.columns.has_duplicates:
        raise ValueError("duplicate feature column names")
    missing = [name for name in IDENTITY_COLUMNS + FEATURE_COLUMNS if name not in frame]
    if missing:
        raise ValueError("missing required feature columns: " + ", ".join(missing))
    if "ddg_exp" in frame:
        raise ValueError("experimental labels belong only to benchmark evaluation")
    if any(str(col).startswith("rankscore_") for col in frame):
        raise ValueError("precomputed rankscore columns are not accepted by this raw-feature interface")
    if any(token in str(col).lower() for col in frame for token in ("foldx", "rosetta", "cartddg")):
        raise ValueError("baseline prediction columns are not accepted by this interface")

    out = frame.copy()
    for name in IDENTITY_COLUMNS:
        values = out[name]
        if values.isna().any() or not values.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} must contain nonempty text identifiers")
        out[name] = values.map(str.strip)
    if out["target"].nunique() != 1:
        raise ValueError("score one target cohort at a time")
    if out["parent_id"].nunique() != 1:
        raise ValueError("score one parent cohort at a time")
    if out.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("duplicate mutation identity in declared cohort")

    for name in FEATURE_COLUMNS:
        try:
            numeric = pd.to_numeric(out[name], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must contain finite numeric values") from exc
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} must contain finite numeric values")
        if name.startswith("n_") and ((numeric < 0).any() or (numeric != np.floor(numeric)).any()):
            raise ValueError(f"{name} must contain nonnegative integer counts")
        out[name] = numeric
    return out


def score_features(frame: pd.DataFrame) -> ScoreResult:
    """Score one complete raw-feature cohort; lower ranks are more favorable.

    This interface consumes already measured channels. It does not generate
    physics energies or ProteinMPNN probabilities from structures.
    """
    prepared = _validate_features(frame)
    prepared["struct_composite"] = compute_struct_composite(
        prepared["n_iface_contacts_8a"].to_numpy(),
        prepared["n_neighbors_10a"].to_numpy(),
    )
    cfg = PepDDGConfig(
        input_csv="<in-memory>",
        output_csv="<in-memory>",
        anchor_column=None,
        anchor_preference=[],
        baseline_column_for_delta=None,
        write_auxiliary_columns=False,
    )
    scored, details = score_dataframe(prepared, cfg)
    if len(scored) != len(frame) or scored[list(IDENTITY_COLUMNS)].reset_index(drop=True).ne(
        prepared[list(IDENTITY_COLUMNS)].reset_index(drop=True)
    ).to_numpy().any():
        raise RuntimeError("scoring did not preserve cohort row identities")

    payload = prepared[list(IDENTITY_COLUMNS) + list(FEATURE_COLUMNS)].to_json(
        orient="records", double_precision=15
    )
    summary = {
        "status": "ok",
        "method": "pepddg_zs",
        "score_kind": "relative_cohort_rank",
        "lower_is_better": True,
        "target": prepared["target"].iloc[0],
        "parent_id": prepared["parent_id"].iloc[0],
        "n_rows": len(scored),
        "input_sha256": sha256(payload.encode("utf-8")).hexdigest(),
        "anchor_source": details["anchor_source"],
        "score_column": cfg.score_column,
    }
    return ScoreResult(scored, summary)


def score_feature_csv(input_csv: str | Path, output_csv: str | Path) -> dict[str, Any]:
    """Validate, score, then atomically publish CSV and adjacent summary JSON."""
    source = Path(input_csv)
    destination = Path(output_csv)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output CSV paths must differ")
    result = score_features(pd.read_csv(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pepddg-", suffix=".csv", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            result.table.to_csv(handle, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    summary_path = destination.with_suffix(destination.suffix + ".json")
    summary_path.write_text(json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result.summary
