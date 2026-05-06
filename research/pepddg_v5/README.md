# PepDDG v19 Sealed Project

PepDDG v19 is sealed around the recovered strict clean-3 lineage.

## Canonical Numbers

- Main common-coverage split: rho = 0.691, 95% CI [0.612, 0.767], N = 332, targets = 33.
- BPTI DMS validation: rho = 0.712, N = 456, targets = 2.
- Four-target OOD DMS validation: rho = 0.235, N = 745, targets = 4.

Source of truth:
`research/pepddg_v5/results/v19_sanitized_baseline/paper_numbers_v19.json`.

## Active Surface

- `results/v19_sanitized_baseline/`: canonical v19 CSV/JSON evidence.
- `results/zs_cal_neurips_release/`: active ZS+Cal release manifests and headline numbers.
- `data/independent_validation/`: BPTI + 3 OOD targets (CC-BY-4.0; see `data/DATA_PROVENANCE.md`).
- `scripts/`: release/reproducibility scripts and statistical-test helpers.
- `internal_tools/pepddg/`: lightweight reusable scoring package.

## Audit

```bash
python -m internal_tools.pepddg.release_audit --repo-root . --json
```
