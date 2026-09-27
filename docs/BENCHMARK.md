# SKEMPI reproduction contract and status

PepDDG's paper result must be checked from molecular inputs, not just from
shipped scores. This repository contains an older frozen SKEMPI-derived
**332-mutation / 33-target** common-coverage feature table. Its recorded
PepDDG-ZS pooled Spearman rho is `0.6188807648016849` (95% interval
`0.5192060230325495`–`0.7078959993425558`). A later rebuttal all-method
common-coverage cohort has **331 mutations / 33 targets** and recorded
PepDDG rho `0.6160850301058195`. An archived three-channel comparison has
**340 mutations / 34 targets**. These are different cohort identities.
The paper's 332/33 main result uses PDB crystal structures; Boltz-2 predicted
wild-type complexes are a separate inference-time diagnostic.

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
prepared crystal structures, executed inputs or complete per-restart
provenance, so B1b does not authenticate a fresh structural run. The original
selected Boltz-2 coordinates for the separate predicted-input diagnostic are
also unavailable.

For source data, obtain the 2018 [SKEMPI 2.0 CSV and cleaned PDB archive](https://life.bsc.es/pid/skempi2/database/index)
from its official download page and review the database's
[CC BY 4.0 terms](https://life.bsc.es/pid/skempi2/info/terms). Keep the
download version, file checksums and attribution with any derived benchmark.
These files are the broad source database, **not** the paper's exact 332/33 or
331/33 cohort. The cleaned PDBs are not authenticated as the exact prepared
crystal inputs used for the main result, and they are not the original Boltz-2
predicted complexes from the separate diagnostic. A verified source-row and
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
