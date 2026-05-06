# PepDDG Internal Tool

Production `PepDDG-ZS` scoring for peptide-protein binding affinity change
(ddG) prediction. The default path is zero-shot and training-free; the frozen
v19 calibrated scorer is retained as explicit `mode: cal` for ablation and
historical comparison.

## What it does

Given a CSV of precomputed per-mutation features, PepDDG applies deterministic
3-channel rank fusion to produce `rankscore_pepddg_zs` by default. No model is
trained or fitted at scoring time.

**Channels** (information sources):
1. **Physics** — OpenMM interaction energy deltas (xint_iface, bind_proxy)
2. **Structure** — Boltz-2 structural confidence metrics
3. **Evolutionary** — ProteinMPNN log-likelihood ratios

The tool does NOT run any heavy computation. It consumes precomputed features and
applies deterministic rank arithmetic. No GPU required, runs in seconds.

## Relationship to research code

- `PepDDG-ZS` is the active manuscript/production algorithm and writes
  `rankscore_pepddg_zs`.
- The active release numbers are generated from
  `research/pepddg_v5/results/zs_cal_neurips_release/paper_numbers_zs_cal.json`.
- `PepDDG-Cal` is the frozen v19 calibrated ablation. It achieved rho=0.691 on
  the sealed common-coverage PepDDG-Bench evaluation (332 mutations across 33
  targets), but must be requested with `mode: cal`.

## Quick start

### CLI usage

```bash
conda activate research

python internal_tools/pepddg/run_pepddg.py \
  --config internal_tools/pepddg/configs/default.yaml \
  --input-csv path/to/features.csv \
  --output-csv path/to/scored.csv
```

Preferred module entrypoint:

```bash
python -m internal_tools.pepddg \
  --config internal_tools/pepddg/configs/default.yaml \
  --input-csv path/to/features.csv \
  --output-csv path/to/scored.csv
```

### Programmatic usage

```python
from internal_tools.pepddg import PepDDGConfig, load_config, run_pepddg

# Option A: from YAML config
cfg = load_config("internal_tools/pepddg/configs/default.yaml")

# Option B: construct directly
cfg = PepDDGConfig(
    input_csv="path/to/features.csv",
    output_csv="path/to/scored.csv",
)

summary = run_pepddg(cfg)
print(summary["status"])        # "ok"
print(summary["score_column"])  # "rankscore_pepddg_zs"
```

### Minimal reproducible example

```bash
cd internal_tools/pepddg/minimal_example
bash run.sh
# Creates output/ with scored CSV + JSON artifacts
```

See `minimal_example/` for a self-contained demo with synthetic data.

## Input CSV format

### Required columns

If `rankscore_3view_base` is present, ZS can score directly from that clean
3-view base column.

| Column | Type | Description |
|--------|------|-------------|
| `rankscore_3view_base` | float | Clean physics + structure + MPNN base score |

If `rankscore_3view_base` is absent, ZS derives the clean 3-view score from
channel columns:

| Column | Type | Description |
|--------|------|-------------|
| `mpnn_neg_llr_complex` | float | ProteinMPNN negative log-likelihood ratio (complex) |
| `mpnn_ddg_bind` | float | ProteinMPNN binding ddG proxy |

### Anchor columns (at least one recommended)

The default ZS mode resolves an anchor score using `rankscore_3view_base`, then
derives a clean 3-view score if needed.

| Column | Priority | Description |
|--------|----------|-------------|
| `rankscore_3view_base` | 1st (default) | Clean 3-view rank score |

Cal-only anchors such as `rankscore_3view_v17_mainboost` and
`rankscore_3view_v17_2_oodguard` are rejected in `mode: zs`; use `mode: cal`
when intentionally reproducing frozen v19 behavior.

### Derived anchor fallback columns

When deriving anchor from components, the tool needs:

| Channel | Column options (tried in order) |
|---------|-------------------------------|
| Physics | `rankscore_phys`, or (`ddg_paired_xint_iface` + `ddg_paired_bind_proxy`) |
| Structure | `rankscore_struct`, `rankscore_struct_base`, `struct_composite` |
| MPNN | `rankscore_mpnn`, or (`mpnn_neg_llr_complex` + `mpnn_ddg_bind`) |

### Optional columns

| Column | Description |
|--------|-------------|
| `ddg_exp` | Experimental ddG (enables gate metrics in output) |
| `target` | Target identifier (for grouping/reporting) |
| `mutation` | Mutation identifier |

## Output files

All outputs are written to `output_dir` (defaults to parent of `output_csv`):

| File | Description |
|------|-------------|
| `<output_csv>` | Input CSV plus score column and optional auxiliary columns |
| `policy_audit_pepddg.json` | Banned-pattern check on input/used columns (clean-3 compliance) |
| `gate_metrics_pepddg.json` | Correlation metrics vs `ddg_exp` (if present) |
| `run_manifest_pepddg.json` | Full config fingerprint + artifact paths for reproducibility |
| `run_summary_pepddg.json` | Compact run status and metadata |

### Scored CSV additional columns

When `write_auxiliary_columns: true` (default), these columns are appended:

| Column | Description |
|--------|-------------|
| `rankscore_pepddg_zs` | Default ZS final score (normalized rank, 0=most stabilizing) |
| `pepddg_mode` | `zs` or `cal` |
| `pepddg_anchor_source` | Which anchor column was used |
| `pepddg_anchor_rank` | Normalized rank of the anchor |
| `pepddg_mpnn_base_rank` | MPNN baseline rank, present only when derived or Cal scoring used it |
| `pepddg_mpnn_neg_rank` | Rank of `mpnn_neg_llr_complex` |
| `pepddg_mpnn_bind_rank` | Rank of `mpnn_ddg_bind` |
| `pepddg_mpnn_disp_rank` | Rank of MPNN dispersion (|neg - bind|) |
| `pepddg_v19_delta_mpnn` | Weighted MPNN residual |
| `pepddg_v19_raw` | Pre-ranking raw score |

## Scoring formula

Default ZS formula:

```
rankscore_pepddg_zs = stable_rank(rankscore_3view_base)
```

If no clean base column exists:

```
phys   = stable_rank(rankscore_phys)
struct = stable_rank(rankscore_struct or rankscore_struct_base or struct_composite)
mpnn   = stable_rank(rankscore_mpnn)
rankscore_pepddg_zs = stable_rank(phys + struct + mpnn)
```

Explicit Cal formula (`mode: cal`) reproduces the frozen v19 residual:

```
delta_mpnn = w_neg*(rank(mpnn_neg) - mpnn_base)
           + w_bind*(rank(mpnn_bind) - mpnn_base)
           - |w_disp_pen|*rank(|mpnn_neg - bind|)
rankscore_3view_v19_strict3 = stable_rank(anchor_rank + w_m * delta_mpnn)
```

## Configuration

Use `configs/default.yaml` as a template. Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `input_csv` | *(required)* | Path to input feature CSV |
| `output_csv` | *(required)* | Path for scored output CSV |
| `output_dir` | parent of output_csv | Directory for JSON artifacts |
| `mode` | `zs` | `zs` default or explicit `cal` |
| `score_column` | `rankscore_pepddg_zs` | Name of the output score column |
| `anchor_column` | `rankscore_3view_base` | Preferred ZS anchor column |
| `anchor_preference` | [`rankscore_3view_base`] | Fallback anchor column list |
| `strict_anchor` | `false` | If true, error when no anchor found |
| `baseline_column_for_delta` | `rankscore_3view_base` | Baseline for delta metrics |
| `write_auxiliary_columns` | `true` | Write intermediate scoring columns |
| `write_run_manifest` | `true` | Write run manifest JSON |
| `write_policy_audit` | `true` | Write policy audit JSON |
| `write_gate_metrics` | `true` | Write gate metrics JSON |
| `variant_name` | `pepddg_zs` | Variant name |
| `banned_patterns` | [foldx, rosetta, ...] | Patterns that fail policy audit |
| `allowed_output_roots` | `[]` | Optional output path confinement |

## Policy audit

The tool enforces a **clean-3 policy**: only physics, structure, and MPNN channels
are allowed. If any input column matches a banned pattern (foldx, rosetta, cartddg,
stabddg, esm2, esmif, esm3, saprot, diffaffinity), the policy audit flags it.

## Dependencies

- Python 3.12+
- `conda activate research`
- Required packages: `numpy`, `scipy`, `pandas`, `pyyaml`
- No GPU, no external model weights, no network access

## Testing

```bash
~/miniconda3/bin/conda run --no-capture-output -n py312 pytest -q \
  unit_tests/internal_tools/test_pepddg_tool.py \
  unit_tests/internal_tools/test_pepddg_release_audit.py
```

## Release audit

```bash
python -m internal_tools.pepddg.release_audit --repo-root . --json
```

The audit recomputes active ZS+Cal metrics from source CSVs, verifies checksums
and manifests, and still checks the historical v19 archive with
`--legacy-v19`.
