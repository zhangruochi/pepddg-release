# PepDDG

**A zero-shot, training-free predictor for peptide–protein binding ΔΔG that
fuses three orthogonal information channels — physical perturbation, geometric
environment, and evolutionary compatibility — by Borda rank aggregation.**

This repository contains the full algorithm implementation, frozen evaluation
bundles, and reproducibility scripts for the headline results of the
accompanying anonymous submission. The manuscript itself is submitted
separately and is **not** distributed in this repository.

| Split | N (mut × targets) | PepDDG-ZS ρ (95% CI) | PepDDG-Cal ρ (95% CI) |
|-------|------------------:|----------------------:|----------------------:|
| Main common-coverage | 332 × 33 | **0.619** [0.519, 0.708] | **0.691** [0.612, 0.770] |
| BPTI control DMS     | 456 × 2  | 0.771 | 0.712 |
| OOD pooled DMS       | 745 × 4  | 0.235 | 0.235 |

PepDDG-ZS is the active zero-shot method; PepDDG-Cal is a calibrated ablation
(weights frozen on a held-out split, no training step at evaluation time).

---

## Repository layout

```
internal_tools/pepddg/        Core algorithm package (importable as a Python module)
  ├── configs/default.yaml    Default rank-fusion config (mode: zs)
  ├── minimal_example/        Self-contained 2-minute smoke test
  ├── test_fixtures/          Tiny synthetic structures and CSVs for tests
  ├── pipeline.py             End-to-end CLI entry point
  ├── feature_assembly.py     Channel merging
  ├── physics_features.py     Physics rank-score extraction
  ├── structural_features.py  Geometric feature derivation
  ├── mpnn_features.py        ProteinMPNN log-likelihood handling
  ├── scoring.py              Rank fusion (Borda)
  ├── release_audit.py        SHA256 verification of released artifacts
  └── ...
research/pepddg_v5/
  ├── results/                Frozen evaluation CSVs and paper number JSONs
  │   ├── v19_sanitized_baseline/   ← per-split eval CSVs (main, BPTI, OOD)
  │   └── zs_cal_neurips_release/   ← canonical headline numbers
  ├── data/independent_validation/  BPTI + 3 OOD targets (CC-BY-4.0; see DATA_PROVENANCE.md)
  ├── scripts/                Analysis, ablation, and statistical-test scripts
  ├── README.md               Project-level overview
  ├── RESULTS_SUMMARY.md      Mode evaluation summary
  ├── NEGATIVE_RESULTS.md     Documented dead ends
  └── LITERATURE_BASELINES.md Comparison protocol
unit_tests/pepddg/
  └── test_v23_token_mapping.py
```

---

## Quickstart

### Environment

```bash
conda env create -f environment.yaml
conda activate pepddg
```

The environment is intentionally lightweight: only NumPy / SciPy / Pandas /
Matplotlib / Seaborn / statsmodels / freesasa / gemmi / openpyxl / pytest are
required. **No GPU, OpenMM, or PyRosetta is needed for the released scoring
pipeline** — the heavy feature-extraction steps were run upstream and their
outputs are frozen in `research/pepddg_v5/results/v19_sanitized_baseline/`.

### (a) Smoke test — ~2 minutes, CPU only

```bash
bash internal_tools/pepddg/minimal_example/run.sh
```

Reads 10 synthetic mutations, runs the rank-fusion scorer, and writes
`internal_tools/pepddg/minimal_example/output/scored.csv`.

### (b) Reproduce the headline numbers — ~5 minutes, CPU only

The CLI re-derives the ZS rank-fusion column from the frozen channel scores in
`main_eval_v19.csv`:

```bash
# Re-derive PepDDG-ZS from channel scores (mode is configured in default.yaml)
python -m internal_tools.pepddg \
  --config internal_tools/pepddg/configs/default.yaml \
  --input-csv research/pepddg_v5/results/v19_sanitized_baseline/main_eval_v19.csv \
  --output-csv /tmp/pepddg_main_zs.csv
```

The PepDDG-Cal column (`rankscore_3view_v19_strict3`) is the result of an
offline calibration step and is shipped pre-computed in `main_eval_v19.csv`.
Verify both rank-correlations against the canonical paper numbers (target:
ZS ≈ 0.619, Cal ≈ 0.691; tolerance ±0.005):

```bash
python - <<'PY'
import json
import pandas as pd
from scipy.stats import spearmanr

gold_path = "research/pepddg_v5/results/zs_cal_neurips_release/paper_numbers_zs_cal.json"
gold = json.load(open(gold_path))

zs_re = pd.read_csv("/tmp/pepddg_main_zs.csv")
src   = pd.read_csv("research/pepddg_v5/results/v19_sanitized_baseline/main_eval_v19.csv")

zs_rho,  _ = spearmanr(zs_re["ddg_exp"], zs_re["rankscore_pepddg_zs"])
cal_rho, _ = spearmanr(src["ddg_exp"],   src["rankscore_3view_v19_strict3"])

print(f"PepDDG-ZS  reproduced rho = {zs_rho:.4f}  (target ~0.619)")
print(f"PepDDG-Cal cached     rho = {cal_rho:.4f} (target ~0.691)")
PY
```

### (c) Audit released artifacts — ~1 minute

Verifies SHA256 hashes of every shipped CSV and JSON manifest:

```bash
python -m internal_tools.pepddg.release_audit --repo-root . --json
```

Status `"PASS"` indicates all hashes match the recorded release manifest.

### (d) Run unit tests

```bash
pytest unit_tests/pepddg -v
```

---

## Citation

This work is currently under anonymous review. Citation details will be added
upon publication.

---

## License

- **Code** (everything under `internal_tools/`, `research/pepddg_v5/scripts/`,
  `unit_tests/`): MIT — see `LICENSE`.
- **Data**: see `research/pepddg_v5/data/DATA_PROVENANCE.md` for per-file
  source, license, and attribution. The released benchmarks are derived from
  SKEMPI v2 (CC-BY-4.0) and the BPTI deep mutational scan of Heyne et al. 2021
  (CC-BY-4.0); please cite those original works alongside this one.
