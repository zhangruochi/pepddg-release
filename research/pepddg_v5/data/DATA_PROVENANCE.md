# Data Provenance & License

This document records the source, license, and redistribution status of every
data file shipped under `research/pepddg_v5/data/` and the sealed evaluation
bundles under `research/pepddg_v5/results/`. Every file in the table is either
(a) a public dataset with a redistributable license, (b) a derived artifact
where the license of the underlying source permits redistribution with
attribution, or (c) a self-generated artifact (PDB structures regenerated from
public coordinates).

If you are a downstream user, please cite the original sources listed below in
addition to this work.

---

## 1. Independent validation — `data/independent_validation/`

### 1.1 BPTI deep mutational scan (Heyne et al. 2021)

| File | Description | Source |
|------|-------------|--------|
| `bpti_dms_heyne2021.xlsx` | Original supplementary spreadsheet from Heyne et al. 2021 | Heyne, M. et al., *Massively Parallel Identification of Single-Domain Antibody Recognition Sites…*, **eLife** 10:e64898 (2021). DOI: 10.7554/eLife.64898 |
| `bpti_dms_single_mutations.csv` | Single-mutant subset reformatted for evaluation | Derived from the above spreadsheet |
| `bpti_cohort.csv` | BPTI cohort definitions used for pooled analyses | Derived |

- **License of original**: CC-BY-4.0 (eLife open-access supplementary).
- **Redistribution**: Allowed with attribution to Heyne et al. 2021.
- **Action in this repo**: shipped as-is for `xlsx`, with derived CSVs alongside;
  attribution kept in this file and (for the pepddg-release tag) referenced from
  the paper supplementary section.

### 1.2 Out-of-distribution targets — DMS data

| File | Target | PDB | Source DMS publication |
|------|--------|-----|------------------------|
| `1CBW_mutations.csv` | Trypsin / BPTI complex (1CBW) | PDB ID 1CBW | Same DMS source (Heyne 2021) |
| `2R9P_mutations.csv` | Trypsin / BPTI variant (2R9P) | PDB ID 2R9P | Same DMS source (Heyne 2021) |
| `3OTJ_mutations.csv` | Trypsin / BPTI variant (3OTJ) | PDB ID 3OTJ | Same DMS source (Heyne 2021) |

- **License**: CC-BY-4.0 (derived from the same Heyne 2021 dataset).
- **Redistribution**: Allowed with attribution.

### 1.3 Crystal structures — `data/independent_validation/crystal_structures/`

| File | Source | License |
|------|--------|---------|
| `1CBW.pdb`, `2R9P.pdb`, `3OTJ.pdb` | RCSB Protein Data Bank | CC0 / public domain |

- **Redistribution**: Allowed without restriction (PDB policy).

### 1.4 Scoring inputs — `data/independent_validation/scoring_input/`

Per-target prepared PDBs and mutation CSVs derived from the above public
sources. License inherited from the originals (CC-BY-4.0 / CC0); redistribution
allowed with attribution.

### 1.5 MPNN inputs — `data/independent_validation/mpnn_input/`

WT structures used as input to ProteinMPNN. Each is a stripped/standardized
copy of the corresponding PDB structure listed in §1.3. License: same as PDB
(CC0).

---

## 2. Sealed evaluation bundles — `results/v19_sanitized_baseline/` and `results/zs_cal_neurips_release/`

### 2.1 Mutation tables and CSVs

| File | Content | Underlying source |
|------|---------|---------------------|
| `main_eval_v19.csv` | 332 SKEMPI v2 peptide–protein mutations × 22 columns of pre-computed features and rank scores | SKEMPI v2 (Jankauskaitė et al. 2019) |
| `bpti_eval_v19.csv` | BPTI control evaluation (456 mutations × 22 columns) | Heyne 2021 (see §1.1) |
| `ood_eval_v19.csv` | OOD pooled evaluation (745 mutations × 22 columns) | Heyne 2021 (see §1.1) |

- **SKEMPI v2 license**: CC-BY-4.0 (https://life.bsc.es/pid/skempi2).
- **Citation**: Jankauskaitė, J., Jiménez-García, B., Dapkūnas, J. et al.,
  *SKEMPI 2.0: an updated benchmark of changes in protein–protein binding
  energy, kinetics and thermodynamics upon mutation*, **Bioinformatics** 35,
  462–469 (2019). DOI: 10.1093/bioinformatics/bty635
- **Redistribution**: Allowed with attribution. The CSVs in this repo are
  derivative works (we ran physics, structural, and MPNN feature extraction on
  the published mutation set) and inherit CC-BY-4.0.

### 2.2 JSON manifests

| File | Content | License |
|------|---------|---------|
| `paper_numbers_v19.json` | Frozen paper numbers (rho, CIs, bootstrap stats) | Original snapshot — MIT (see `LICENSE-MIT`) |
| `paper_numbers_zs_cal.json` | ZS+Cal release numbers | This work — MIT |
| `recovery_manifest_v19.json` | SHA256 manifest for reproducibility audit | This work — MIT |
| `release_manifest_zs_cal.json` | Same, for ZS+Cal release | This work — MIT |
| `v19_release_manifest.json` | Top-level pointer manifest | This work — MIT |

- **No external data** is embedded in the JSON manifests beyond cryptographic
  hashes of the CSVs already covered above.

---

## 3. Algorithm code

The original snapshot at commit `4fae29b7bb9f5cf7befcb50b8badc392a848864c`
was released under MIT; its original code retains that grant (see
`LICENSE-MIT`). New packaging, structural orchestration and tests in later
commits are under PolyForm Noncommercial 1.0.0 (see top-level `LICENSE` and
`LICENSE_SCOPE.md`). Directory names alone do not determine the terms of a
particular file or contribution.

---

## 4. Attribution recap (please cite if you use this resource)

1. **PepDDG (this work)** — citation TBD (NeurIPS submission, currently anonymous).
2. **SKEMPI v2** — Jankauskaitė et al., *Bioinformatics* (2019).
3. **BPTI DMS** — Heyne et al., *eLife* (2021).
4. **Crystal structures** — RCSB PDB.

---

## 5. Files explicitly NOT redistributed

- Per-restart OpenMM energy sidecars: not redistributed because the canonical
  pooled energetic columns are already present in `main_eval_v19.csv`.
  Reproducing the per-restart ablation (`scripts/sensitivity_energetic.py`)
  requires the user to set `PEPDDG_RESTARTS_ROOT` to a local cache directory
  containing the per-restart JSONs; the baseline scoring path does NOT depend
  on this.
