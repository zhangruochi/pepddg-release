#!/usr/bin/env python
"""CLI for PepDDG scoring."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import sys
from pathlib import Path

# Ensure repo root on sys.path for script execution mode.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from internal_tools.pepddg.config import load_config
from internal_tools.pepddg.pipeline import run_pepddg


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pepddg.run_pepddg")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PepDDG scoring.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--input-csv", default=None, help="Optional override for input_csv.")
    parser.add_argument("--output-csv", default=None, help="Optional override for output_csv.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)

        if args.input_csv:
            cfg = replace(cfg, input_csv=str(args.input_csv))
            cfg.validate()
        if args.output_csv:
            out_csv = Path(args.output_csv)
            cfg = replace(cfg, output_csv=str(out_csv), output_dir=str(out_csv.parent))
            cfg.validate()

        summary = run_pepddg(cfg)
        logger.info("PepDDG finished: %s", summary)
        return 0
    except Exception:
        logger.exception("PepDDG failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
