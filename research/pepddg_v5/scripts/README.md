# PepDDG Scripts

This directory contains release/reproducibility scripts and helpers used by the
ablation and statistical-significance analyses. Use `internal_tools/pepddg/` as
the entry point for new scoring work; do not add new production entrypoints
here unless they are reproduction scripts for the released numbers.

The `v*_common.py` files are utility modules implementing the channel-fusion
score variants (v10 through v18) referenced by the assemble/optimize scripts.
