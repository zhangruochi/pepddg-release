from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PepDDGConfig
from .scoring import find_banned_columns, score_dataframe


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _config_sha256(cfg: PepDDGConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_pepddg(cfg: PepDDGConfig) -> dict[str, Any]:
    """Run PepDDG scoring pipeline on one input CSV."""
    cfg.validate()
    in_path = Path(cfg.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_csv = Path(cfg.output_csv)
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    input_columns = list(pd.read_csv(in_path, nrows=0).columns)
    df = pd.read_csv(in_path)
    scored, details = score_dataframe(df, cfg)
    scored.to_csv(out_csv, index=False)

    policy = {
        "mode": cfg.mode,
        "variant": cfg.variant_name,
        "score_column": cfg.score_column,
        "anchor_source": details["anchor_source"],
        "anchor_used_columns": details["anchor_used_columns"],
        "input_columns_raw": sorted(input_columns),
        "input_columns_loaded": sorted(list(df.columns)),
        "used_columns": details["used_columns"],
        "strict_clean3_columns_used": details["used_columns"],
        "banned_patterns": cfg.banned_patterns,
    }
    policy["input_columns_with_banned_patterns_raw"] = find_banned_columns(
        policy["input_columns_raw"], cfg.banned_patterns
    )
    policy["input_columns_with_banned_patterns_loaded"] = find_banned_columns(
        policy["input_columns_loaded"], cfg.banned_patterns
    )
    policy["used_columns_violations"] = find_banned_columns(
        policy["used_columns"], cfg.banned_patterns
    )
    policy["pass"] = bool(
        len(policy["input_columns_with_banned_patterns_raw"]) == 0
        and len(policy["input_columns_with_banned_patterns_loaded"]) == 0
        and len(policy["used_columns_violations"]) == 0
    )

    gate_metrics = dict(details["gate_metrics"])
    gate_metrics["mode"] = cfg.mode
    gate_metrics["score_column"] = cfg.score_column
    gate_metrics["anchor_source"] = details["anchor_source"]
    gate_metrics["n_rows"] = int(len(scored))

    summary = {
        "status": "ok",
        "mode": cfg.mode,
        "variant": cfg.variant_name,
        "created_utc": _utc_now(),
        "input_csv": str(in_path.resolve()),
        "output_csv": str(out_csv.resolve()),
        "output_dir": str(out_dir.resolve()),
        "n_rows": int(len(scored)),
        "anchor_source": details["anchor_source"],
        "score_column": cfg.score_column,
        "policy_pass": bool(policy["pass"]),
        "gate_metrics_available": bool(gate_metrics.get("available", False)),
    }

    if cfg.write_policy_audit:
        _write_json(out_dir / "policy_audit_pepddg.json", policy)
    if cfg.write_gate_metrics:
        _write_json(out_dir / "gate_metrics_pepddg.json", gate_metrics)
    if cfg.write_run_manifest:
        manifest = {
            "created_utc": summary["created_utc"],
            "mode": cfg.mode,
            "variant": cfg.variant_name,
            "config_sha256": _config_sha256(cfg),
            "config": asdict(cfg),
            "artifacts": {
                "output_csv": str(out_csv.resolve()),
                "policy_audit": str((out_dir / "policy_audit_pepddg.json").resolve()),
                "gate_metrics": str((out_dir / "gate_metrics_pepddg.json").resolve()),
            },
            "provenance": {
                "mode": cfg.mode,
                "score_column": cfg.score_column,
                "anchor_source": details["anchor_source"],
                "anchor_used_columns": details["anchor_used_columns"],
                "used_columns": details["used_columns"],
            },
            "summary": summary,
        }
        _write_json(out_dir / "run_manifest_pepddg.json", manifest)

    _write_json(out_dir / "run_summary_pepddg.json", summary)
    return summary
