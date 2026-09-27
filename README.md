# PepDDG

PepDDG ranks amino-acid substitutions in a peptide–protein complex using
physical, structural and ProteinMPNN-derived channels. Lower PepDDG-ZS scores
mean more favorable *relative to the mutation cohort supplied for the same
parent and target*. They are not absolute binding free energies, experimental
affinities or evidence of wet-lab binding.

The accompanying paper, **“PepDDG: Peptide–Protein Binding ΔΔG Prediction via
Information Channel Decomposition,”** was accepted to NeurIPS 2026. The
official author list and proceedings identifier are not yet verified in this
repository; see [citation guidance](#citation).

## Install

From a clone of this repository, create the complete CPU-capable environment
with one command:

```bash
conda env create -f environment.yaml
conda activate pepddg
pepddg --help
pepddg doctor --json
```

`environment.yaml` installs OpenMM and PDBFixer from conda-forge, PyTorch,
ProteinMPNN's bundled v_48_020 checkpoint and this package. The structural
workflow defaults to CPU; CUDA needs a compatible OpenMM/CUDA installation and
your own compute resources. The lightweight feature-only path can instead be
installed with `python -m pip install .` in an existing Python 3.12 environment.
The full structural workflow needs the conda environment because PDBFixer is
distributed through conda-forge. See [installation and input details](docs/STRUCTURES.md).
`pepddg doctor` checks dependency and bundled-checkpoint presence without
running a model; presence alone does not establish a working OpenMM platform.
For a concise Chinese walkthrough, see [中文快速入门](docs/QUICKSTART.zh-CN.md).

## Use

To score a complete table of previously generated raw channels:

```bash
pepddg score-features --input examples/raw_features.csv --output /tmp/pepddg-scores.csv
```

The output contains the original row identities and PepDDG ranks; an adjacent
JSON file records the cohort and input hash. Required columns are `target`,
`parent_id`, `mutation`, `ddg_xint_iface`, `ddg_bind_proxy`,
`n_iface_contacts_8a`, `n_neighbors_10a`, `mpnn_neg_llr_complex` and
`mpnn_ddg_bind`. The interface rejects missing/nonfinite channels, repeated
mutation identities, mixed target/parent cohorts, experimental labels and
precomputed ranks and any extra columns. **A feature CSV is not an end-to-end reproduction.**

To generate these channels from a prepared linear complex and rank its
mutations:

```bash
pepddg score-structures \
  --structure /path/to/linear-complex.pdb \
  --peptide-chain P --receptor-chain R \
  --mutations /path/to/mutations.csv \
  --target target_001 --parent-id WT \
  --output /tmp/pepddg-structural-result
```

The default is seven paired OpenMM restarts for each mutation, with WT and
mutant using the same mutation-site restraint exclusion. This can be
slow on CPU. This command is a **workflow template**, not the paper's
full SKEMPI cohort or a validated prediction for cyclic peptides. The
single-receptor-chain input contract and currently unsupported chemistries are
documented in [structural workflow](docs/STRUCTURES.md). Successful runs write
`features.csv`, `scores.csv` and `provenance.json` only after every requested
mutation has all required channels.

Python users can call `pepddg.score_features(frame)` or
`pepddg.run_structural_cohort(spec, target=..., parent_id=...,
output_dir=...)`. The public input types are `ComplexSpec` and `MutationSpec`.
See [API examples](docs/STRUCTURES.md).

## Reproducibility and evidence

The historical release includes a frozen 332-mutation, 33-target SKEMPI-derived
feature table and a recorded PepDDG-ZS pooled Spearman correlation near 0.619.
The later rebuttal all-method common-coverage analysis has 331 mutations and
33 targets; the two cohorts must not be conflated. The frozen table can test
rank-fusion regression, but it lacks complete molecular mutation identities
and cannot by itself prove a fresh structure-to-score reproduction. See
[benchmark protocol and current verification status](docs/BENCHMARK.md).

The data files have separate provenance and rights; see
[DATA_PROVENANCE.md](research/pepddg_v5/data/DATA_PROVENANCE.md). The bundled
ProteinMPNN code and checkpoint retain their MIT notice. Source snapshots and
hashes for the structural producer are in
`internal_tools/pepddg/_backend/SOURCE_PROVENANCE.json`.

To replay the published 332-row **frozen feature** result from
an installed package and this clone, run:

```bash
python -m pepddg.frozen_benchmark --repo-root . --output /tmp/pepddg-frozen-replay
```

This writes row-level scores and a metric report. Experimental labels are
kept out of the predictor input. A PASS verifies the recorded score/metric
calculation only; the fresh structural SKEMPI run has a separate gate.

## License and commercial use

New release packaging and structural orchestration are under
[PolyForm Noncommercial 1.0.0](LICENSE). Commercial use of those materials,
including company-internal R&D, needs a separate written authorization.
The original repository snapshot was MIT-licensed and that grant remains in
force for its covered code. Read [license scope](LICENSE_SCOPE.md) before using
the combined edition; it explains third-party and historical rights. For a
commercial license inquiry, contact <zrc720@gmail.com>.

## Citation

Please cite the PepDDG NeurIPS 2026 paper by its verified title above. The
checked manuscript still uses an anonymous author placeholder; the final
authors and DOI/proceedings URL will be added when authenticated. Do not use
“Anonymous Authors” as the post-acceptance citation.

## Development checks

```bash
pytest -q unit_tests/pepddg
python -m pepddg.release_audit --repo-root . --json
```

`release_audit` checks the originally shipped frozen files. Its artifact hash
PASS is distinct from a fresh scientific reproduction. Research-side work can
continue separately; this release changes only through deliberate versioned
updates.

See [release history](CHANGELOG.md), [contribution guidance](CONTRIBUTING.md),
and [third-party notices](THIRD_PARTY_NOTICES.md).
