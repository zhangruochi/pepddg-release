# PepDDG-ZS — AI Agent Workflow Guide

End-to-end workflow for mutation DDG ranking using the production PepDDG-ZS
3-channel scorer. ZS is the default zero-shot/training-free path. Use
`mode: cal` only when intentionally reproducing the frozen v19 calibrated
ablation.

## Pipeline Overview

```
PDB + Mutations
     │
     ├── Step 1: Structure Parsing ──► structure_io.parse_structure()
     │
     ├── Step 2: Physics Features ──► OpenMM implicit scoring (K3s)
     │   └── physics_features.extract_physics_from_csv()
     │
     ├── Step 3: MPNN Features ──► ProteinMPNN inference (K3s)
     │   └── mpnn_features.extract_mpnn_from_csv()
     │
     ├── Step 4: Structural Features ──► structural_features.compute_structural_features_batch()
     │
     ├── Step 5: Assembly ──► feature_assembly.assemble_features()
     │
     └── Step 6: Scoring ──► scoring.score_dataframe() or pipeline.run_pepddg()
```

---

## Step 1: Structure Parsing

**What**: Parse PDB or CIF structure files into atom records.

**Function**: `structure_io.parse_structure(path, chain_ids=None, atom_filter="heavy")`

**Input**: PDB or CIF file (auto-detected by extension)

**Output**: `list[AtomRecord]` — each record has chain, resnum, resname, atom_name, coord, element

**Usage**:
```python
from internal_tools.pepddg.structure_io import parse_structure, get_chain_sequence

atoms = parse_structure("complex.pdb")
seq = get_chain_sequence(atoms, "B")  # {1: 'A', 2: 'G', ...}
```

**Runtime**: Instant (file parsing only)

**Notes**:
- Supports both PDB and CIF (Boltz-2 outputs CIF)
- Hydrogen atoms are excluded by default
- Chain IDs filter at parse time for efficiency

---

## Step 2: Physics Features (OpenMM Implicit Scoring)

**What**: Extract delta energies from pre-computed OpenMM implicit solvent scoring.

**Prerequisites**: Run OpenMM scoring for both WT and mutant structures. This is a K3s job.

**Functions**:
```python
from internal_tools.pepddg.physics_features import (
    extract_physics_from_openmm_results,  # dict-based
    extract_physics_from_csv,             # CSV-based
    compute_physics_rankscore,            # rankscore derivation
)
```

**From OpenMM result dicts** (programmatic):
```python
result = extract_physics_from_openmm_results(wt_result, mut_result)
# Returns: {"ddg_xint_iface": float, "ddg_bind_proxy": float}
```

**From benchmark CSV** (batch):
```python
physics_df = extract_physics_from_csv("results.csv")
# Columns: target, mutation, ddg_xint_iface, ddg_bind_proxy
# Prefers ddg_paired_* columns if available
```

**Energy keys used**:
- `ddg_xint_iface` = Δ(e_cross_interface_total_screened_heavy_kcal_mol)
- `ddg_bind_proxy` = Δ(dg_bind_kcal_mol)

**Runtime**: Extraction is instant. OpenMM scoring itself runs ~5-30 min per mutation on K3s.

**K3s**: Use the `openmm-md` Docker image. Submit indexed jobs for >5 mutations.

---

## Step 3: MPNN Features (ProteinMPNN Scoring)

**What**: Compute log-likelihood ratio features from ProteinMPNN.

**Prerequisites**: Run ProteinMPNN inference for both complex and isolated chain contexts.

**Functions**:
```python
from internal_tools.pepddg.mpnn_features import (
    compute_mpnn_ddg_features,  # single mutation
    extract_mpnn_from_csv,      # batch from CSV
    compute_mpnn_rankscore,     # rankscore derivation
)
```

**Single mutation** (from log-probabilities):
```python
result = compute_mpnn_ddg_features(
    complex_logp_wt=-2.5, complex_logp_mut=-1.8,
    chain_logp_wt=-3.0, chain_logp_mut=-2.0,
)
# Returns: {"mpnn_neg_llr_complex": -0.7, "mpnn_ddg_bind": 0.3}
```

**From CSV** (batch):
```python
mpnn_df = extract_mpnn_from_csv("mpnn_scores.csv")
# Handles two formats:
# 1. Pre-computed: already has mpnn_neg_llr_complex, mpnn_ddg_bind
# 2. Raw log-probs: mpnn_logp_{wt,mut}_{complex,chain} columns
```

**Formulas**:
- `mpnn_neg_llr_complex = -(logP_mut_complex - logP_wt_complex)`
- `mpnn_ddg_bind = -(logP_mut_complex - logP_wt_complex) + (logP_mut_chain - logP_wt_chain)`

**Runtime**: Extraction is instant. MPNN inference runs ~1-5 min per target on K3s.

---

## Step 4: Structural Features

**What**: Compute interface contacts and neighbor counts from the 3D structure.

**Functions**:
```python
from internal_tools.pepddg.structural_features import (
    MutationSite,
    compute_structural_features_batch,
    compute_struct_composite,
)
```

**Batch computation** (recommended):
```python
mutations = [
    MutationSite(chain_id="B", resnum=1, wt_aa="A", mut_aa="V", label="A1V"),
    MutationSite(chain_id="B", resnum=3, wt_aa="G", mut_aa="L", label="G3L"),
]
struct_df = compute_structural_features_batch(
    "complex.pdb", mutations,
    mutation_chain="B", partner_chains=["A"],
)
# Columns: label, n_iface_contacts_8a, n_neighbors_10a
```

**Features**:
- `n_iface_contacts_8a`: Heavy-atom pairs between mutation residue and partner chains within 8Å
- `n_neighbors_10a`: CA atoms (any chain) within 10Å of mutation site CB/CA (CA for GLY)
- `struct_composite`: zscore(contacts) + zscore(neighbors)

**Runtime**: Instant (numpy/scipy distance computation)

---

## Step 5: Feature Assembly

**What**: Merge all three channels and compute derived rankscores.

**Function**:
```python
from internal_tools.pepddg.feature_assembly import assemble_features, validate_scoring_input

assembled = assemble_features(
    physics_df, structural_df, mpnn_df,
    join_on=["target", "mutation"],
    compute_rankscores=True,
)
# Adds: rankscore_phys, struct_composite, rankscore_mpnn

report = validate_scoring_input(assembled)
assert report["valid"], f"Missing columns: {report['missing']}"
```

**Merge behavior**: Inner join on `join_on` keys. Mutations missing from any channel are dropped.

**Derived columns** (when `compute_rankscores=True`):
- `rankscore_phys`: rank(rank(xint) + rank(bind)), normalized [0,1]
- `struct_composite`: zscore(contacts) + zscore(neighbors)
- `rankscore_mpnn`: rank(rank(neg_llr) + rank(ddg_bind)), normalized [0,1]

---

## Step 6: Scoring

**What**: Apply PepDDG-ZS rank-fusion scorer by default.

**Function**: `scoring.score_dataframe(df, cfg)` or `pipeline.run_pepddg(cfg)`

**Required columns**: `rankscore_3view_base`, or enough channel columns to
derive it (`rankscore_phys`, structure rank/composite, and MPNN rank/features).

**Anchor resolution**:
1. Pre-computed ZS base: `rankscore_3view_base`
2. Derived clean-3 base: `stable_rank(phys + struct + mpnn)`

Cal-only anchors such as `rankscore_3view_v17_mainboost` are rejected in ZS
mode. Explicit Cal mode restores the frozen v19 anchor/score defaults.

**Usage with assembled features**:
```python
from internal_tools.pepddg import PepDDGConfig, run_pepddg

assembled.to_csv("/tmp/features.csv", index=False)
cfg = PepDDGConfig(
    input_csv="/tmp/features.csv",
    output_csv="/tmp/scored.csv",
    anchor_column=None,           # use derived anchor
    anchor_preference=[],
    strict_anchor=False,
)
result = run_pepddg(cfg)
```

**Output**: Scored CSV with `rankscore_pepddg_zs` column + JSON artifacts.

---

## Full Example: New Target Mutation Ranking

```python
from internal_tools.pepddg import (
    MutationSite, PepDDGConfig,
    assemble_features, compute_structural_features_batch,
    extract_mpnn_from_csv, extract_physics_from_csv,
    run_pepddg, validate_scoring_input,
)

# 1. Define mutations
mutations = [
    MutationSite("B", 5, "A", "V", "A5V"),
    MutationSite("B", 8, "L", "F", "L8F"),
    MutationSite("B", 12, "G", "A", "G12A"),
]

# 2. Structural features (instant)
struct_df = compute_structural_features_batch(
    "complex.pdb", mutations, "B", ["A"]
)
struct_df["target"] = "my_target"
struct_df["mutation"] = struct_df["label"]

# 3. Physics features (from pre-computed OpenMM results)
phys_df = extract_physics_from_csv("openmm_results.csv")

# 4. MPNN features (from pre-computed MPNN results)
mpnn_df = extract_mpnn_from_csv("mpnn_results.csv")

# 5. Assemble
assembled = assemble_features(phys_df, struct_df, mpnn_df)
report = validate_scoring_input(assembled)
assert report["valid"], f"Missing: {report['missing']}"

# 6. Score
assembled.to_csv("/tmp/features.csv", index=False)
cfg = PepDDGConfig(
    input_csv="/tmp/features.csv",
    output_csv="/tmp/scored.csv",
    anchor_column=None,
    anchor_preference=[],
    strict_anchor=False,
)
result = run_pepddg(cfg)
# result has scored CSV + gate metrics + policy audit
```

For the frozen calibrated ablation:

```python
cfg = PepDDGConfig(
    input_csv="/tmp/features.csv",
    output_csv="/tmp/scored_cal.csv",
    mode="cal",
)
result = run_pepddg(cfg)
# Writes rankscore_3view_v19_strict3
```

---

## K3s Job Patterns

### OpenMM Scoring (Step 2)
```bash
# Use openmm-md Docker image
# Submit indexed job: 1 mutation per pod
# See docs/k3s/ for job templates
```

### ProteinMPNN Scoring (Step 3)
```bash
# Use proteinmpnn Docker image
# Run complex + chain scoring per target
# batch_size=16, timeout=1800s
```

### Batch Pipeline (>10 mutations)
For large-scale scoring, pre-compute Steps 2-3 on K3s, then run Steps 4-6 locally.

---

## Testing

```bash
# Run all PepDDG tests
~/miniconda3/bin/conda run --no-capture-output -n py312 pytest -q \
  unit_tests/internal_tools/test_pepddg_*.py
```
