# SKEMPI reproduction contract and status

## Current four-target smoke

The scoped end-to-end release smoke covers all 35 official SKEMPI v2.0
observations from the four targets labeled CYCLIC in the paper: 1SMF (5),
3EQS (10), 3EQY (11), and 5XCO (9). The downloaded structures show that 1SMF
and 5XCO are disulfide-closed; 3EQS and 3EQY are linear PMI controls. The
reference run completed seven paired restarts and all three channels for each
observation. Its detailed source and runtime evidence is in
[`examples/skempi_cyclic/`](../examples/skempi_cyclic/README.md).

| Group | Mean absolute experimental-rho drift | Mean paper-score order agreement | Result |
|---|---:|---:|---|
| All four paper-defined targets | 0.0806 | 0.9019 | PASS |
| True disulfide pair (1SMF, 5XCO) | 0.0293 | 0.9250 | PASS |

Per-target results also pass the predeclared limits: absolute rho drift ≤0.20
and score-order agreement ≥0.70. The grouped limits are ≤0.10 and ≥0.90.
These are small-cohort engineering tolerances, not confidence intervals or
statistical-equivalence margins. Experimental labels and historical scores are
used only by the evaluator after inference. The new official crystal structures
and cofactor-free receptor preparation differ from unavailable historical
predicted inputs; this smoke therefore checks broad rank consistency, not
bitwise or exact-input reproduction.

To run all four cohorts and regenerate the metric table and chart after
installing the documented environment:

```bash
bash examples/skempi_cyclic/run.sh /path/to/output
```

PepDDG's paper result must be checked from molecular inputs, not just from
shipped scores. This repository contains an older frozen SKEMPI-derived
**332-mutation / 33-target** common-coverage feature table. Its recorded
PepDDG-ZS pooled Spearman rho is `0.6188807648016849` (95% interval
`0.5192060230325495`–`0.7078959993425558`). A later rebuttal all-method
common-coverage cohort has **331 mutations / 33 targets** and recorded
PepDDG rho `0.6160850301058195`. An archived three-channel comparison has
**340 mutations / 34 targets**. These are different cohort identities.
The manuscript describes the 332/33 main result as using PDB crystal
structures and Boltz-2 predicted complexes as a separate diagnostic. However,
the archived phase-0 producer names a predicted-score input, and all 332
frozen-cohort raw physics pairs match the archived predicted-score table while
none match the archived crystal-score table. The executed intermediate,
prepared structures and per-restart records are unavailable, so the main
result's structure provenance is **unresolved**. The frozen replay must not be
presented as a verified crystal-input or fresh structural result.
The source comparison is pinned to platform commit
`999ff87db46afecd293b72cfa61c8e3ae2a21419`:
`research/pepddg_v5/scripts/phase0_baseline.py`,
`results/phase0/cohort_locked.csv`, and the `all_scores_predicted.csv` and
`all_scores_crystal.csv` tables in that research project.

The original 332-row released CSV has no complete molecular mutation keys or
raw ProteinMPNN probabilities. Its row order is suitable for testing the
frozen rank-fusion code; it is insufficient as a fresh structural-run input.
We therefore report five separate checks:

| Check | Required evidence | Current status |
|---|---|
| B0 identity/protocol lock | Exact target–chain–residue–WT–mut keys, structures, exclusions, channel definitions and final metric manifest for all relevant cohorts | In progress; do not infer missing keys from rounded features |
| B1a frozen regression | Installed module recomputes the 332-row score vector, pooled rho and interval without giving labels to the predictor | PASS on the installed wheel: 332/33, rho `0.6188807648016849`, interval `[0.5192060230325495, 0.7078959993425558]`, zero score/metric drift. Re-run with `python -m pepddg.frozen_benchmark --repo-root . --output /tmp/pepddg-frozen-replay` |
| B1b historical channel reconstruction | Archived physics, geometry and ProteinMPNN channel tables independently reproduce all 332 original channel ranks and the fused ZS scores | PASS within the archived-table assembly scope; original per-restart producers and structures remain unauthenticated |
| B2 fresh end-to-end | Complete 332/33 and final-paper cohorts rerun from structures with all channels, seven paired restarts and exact member accounting | Not run |
| B3 platform parity | Installed release and platform adapter exercise real orchestration and agree on keyed fresh results | Not run |

For a B2 PASS, every required mutation must complete all channels and all
seven restart attempts, with no dropped or imputed rows. The new 332-row
PepDDG-ZS pooled rho must be within `0.02` of the historical rho, and the
new per-mutation scores must have Spearman agreement at least `0.95` with
the frozen keyed scores. Final-paper acceptance additionally requires the
authenticated 331/33 cohort and per-target aggregation, coverage and
prioritization metrics. These are prospective engineering tolerances, not a
claim of statistical equivalence. The exact benchmark specification and
rebuttal evidence index live in the platform workflow plan.

The B1a command writes `scores.csv` and `report.json`. It validates input
hashes, the 332/33 membership, the complete ranked vector, pooled rho and
the recorded target-bootstrap interval against tolerances fixed before this
packaging work. The report names its scope `published_anchor_rank_regression_not_structure_to_score`.

The B1b reconstruction uses the historical 455-row normalization population
before selecting the 332 released observations. It reproduced all four rank
columns exactly; the largest difference from the independent B1a ZS score
file was `1.11e-16`. These archived tables do not contain the original
prepared structures, executed inputs or complete per-restart provenance, so
B1b does not authenticate a fresh structural run. The structure-source
conflict above must be resolved before assigning the frozen cohort to either
input route.

For source data, obtain the 2018 [SKEMPI 2.0 CSV and cleaned PDB archive](https://life.bsc.es/pid/skempi2/database/index)
from its official download page and review the database's
[CC BY 4.0 terms](https://life.bsc.es/pid/skempi2/info/terms). Keep the
download version, file checksums and attribution with any derived benchmark.
These files are the broad source database, **not** the paper's exact 332/33 or
331/33 cohort. The cleaned PDBs are not authenticated as the exact prepared
structure inputs used for the main result, and they are not the original
Boltz-2 predicted complexes. A verified source-row and
structure crosswalk is still being recovered; do not select rows by matching
experimental labels. The package's noncommercial code terms and the SKEMPI
data terms apply to their respective materials separately.

The paper's seven-restart OpenMM protocol is expensive. A one-case CPU smoke
tests only execution. The available frozen CSV can be replayed with the
legacy `pepddg --config ...` path, but that check is only B1a when the
recorded row identity, recomputed score vector, rho and bootstrap interval
are all verified. It cannot substitute for B0, B1b, B2 or B3. No current
file in this release should be read as asserting that the structural
workflow has reproduced the paper result.
