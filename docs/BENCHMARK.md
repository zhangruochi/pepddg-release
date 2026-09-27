# SKEMPI reproduction contract and status

PepDDG's paper result must be checked from molecular inputs, not just from
shipped scores. This repository contains an older frozen SKEMPI-derived
**332-mutation / 33-target** common-coverage feature table. Its recorded
PepDDG-ZS pooled Spearman rho is `0.6188807648016849` (95% interval
`0.5192060230325495`–`0.7078959993425558`). A later rebuttal all-method
common-coverage cohort has **331 mutations / 33 targets** and recorded
PepDDG rho `0.6160850301058195`. An archived three-channel comparison has
**340 mutations / 34 targets**. These are different cohort identities.

The original 332-row released CSV has no complete molecular mutation keys or
raw ProteinMPNN probabilities. Its row order is suitable for testing the
frozen rank-fusion code; it is insufficient as a fresh structural-run input.
We therefore report four separate checks:

| Check | Required evidence | Current status |
|---|---|
| B0 identity/protocol lock | Exact target–chain–residue–WT–mut keys, structures, exclusions, channel definitions and final metric manifest for all relevant cohorts | In progress; do not infer missing keys from rounded features |
| B1a frozen regression | Installed CLI recomputes the 332-row score vector, pooled rho and interval without reading the stored aggregate | Pending full recorded check |
| B1b historical raw-channel reconstruction | Historical raw energy/restart, geometry and MPNN terms reproduce original channel ranks | Pending authenticated raw-channel recovery |
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

The paper's seven-restart OpenMM protocol is expensive. A one-case CPU smoke
tests only execution. The available frozen CSV can be replayed with the
legacy `pepddg --config ...` path, but that check is only B1a when the
recorded row identity, recomputed score vector, rho and bootstrap interval
are all verified. It cannot substitute for B0, B1b, B2 or B3. No current
file in this release should be read as asserting that the structural
workflow has reproduced the paper result.
