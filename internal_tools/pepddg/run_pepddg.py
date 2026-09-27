#!/usr/bin/env python
"""CLI for PepDDG scoring."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys

from .config import load_config
from .pipeline import run_pepddg
from .api import score_feature_csv


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "score-features":
        parser = argparse.ArgumentParser(description="Score one complete PepDDG raw-feature cohort.")
        parser.add_argument("--input", required=True, help="Raw feature CSV")
        parser.add_argument("--output", required=True, help="Scored CSV")
        args = parser.parse_args(argv[1:])
        try:
            logger.info("PepDDG scored: %s", score_feature_csv(args.input, args.output))
            return 0
        except Exception:
            logger.exception("PepDDG feature scoring failed.")
            return 2
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
