#!/usr/bin/env python
"""CLI for PepDDG scoring."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys

from .config import load_config
from .pipeline import run_pepddg
from .api import score_feature_csv
from .structure_contract import ComplexSpec, read_mutations_csv
from .structural_pipeline import run_structural_cohort


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
    if argv and argv[0] == "doctor":
        parser = argparse.ArgumentParser(description="Check PepDDG dependencies and bundled resources without scoring.")
        parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
        args = parser.parse_args(argv[1:])
        from .diagnostics import diagnose_runtime

        report = diagnose_runtime()
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print("Feature scoring dependencies present:", report["feature_scoring_ready"])
            print("Structural dependencies and checkpoint present:", report["structural_runtime_ready"])
            for name, detail in report["dependencies"].items():
                print(f"  {name}: {detail['version'] if detail['available'] else 'missing'}")
            print("This checks presence only; it does not run a structural prediction.")
        return 0
    if argv and argv[0] == "score-structures":
        parser = argparse.ArgumentParser(description="Score a linear receptor-peptide mutation cohort from a structure.")
        parser.add_argument("--structure", required=True, help="PDB or mmCIF complex")
        parser.add_argument("--peptide-chain", required=True)
        parser.add_argument("--receptor-chain", required=True)
        parser.add_argument("--mutations", required=True, help="Explicit mutation CSV")
        parser.add_argument("--target", required=True)
        parser.add_argument("--parent-id", required=True)
        parser.add_argument("--output", required=True, help="New or empty output directory")
        parser.add_argument("--closure", default="linear", help="Molecular closure; linear validated, disulfide preview unqualified")
        parser.add_argument("--n-restarts", type=int, default=7)
        parser.add_argument("--seed", type=int, default=20260302)
        parser.add_argument("--platform", choices=("CPU", "CUDA"), default="CPU")
        parser.add_argument("--cpu-threads", type=int, default=2)
        args = parser.parse_args(argv[1:])
        try:
            spec = ComplexSpec(
                args.structure, args.peptide_chain, (args.receptor_chain,),
                read_mutations_csv(args.mutations), args.closure,
            )
            result = run_structural_cohort(
                spec, target=args.target, parent_id=args.parent_id, output_dir=args.output,
                n_restarts=args.n_restarts, seed=args.seed, platform=args.platform,
                cpu_threads=args.cpu_threads,
            )
            logger.info("PepDDG structural cohort scored: %s", result.provenance)
            return 0
        except Exception:
            logger.exception("PepDDG structural scoring failed.")
            return 2
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
