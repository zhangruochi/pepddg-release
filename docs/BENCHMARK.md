# SKEMPI reproduction contract and status

## Current release acceptance

The release test uses the paper-defined CYCLIC subset: **35 mutations across four targets**, obtained from the official SKEMPI v2 CSV and cleaned PDB archive. All observation identities, affinities, cleaned/author numbering and WT residues are checked before inference. See [the executable example](../examples/skempi_cyclic/README.md) for download/input hashes, mutations and prepared coordinates.

| Target | Observations | Actual chemistry |
|---|---:|---|
| 1SMF | 5 | Disulfide |
| 5XCO | 9 | Disulfide |
| 3EQS | 10 | Linear PMI control |
| 3EQY | 11 | Linear PMI control |

3EQS/3EQY carry the paper's CYCLIC annotation but do not have covalent ring closure. Acceptance reports the complete35/4 population and separately the actual disulfide14/2 subgroup. No terminal bond is invented and no failing row is dropped.

Fresh execution must generate all six raw features for every mutation, seven paired WT/mutant OpenMM restarts, and pre/post closure and stereochemistry evidence. Restart arrays and structures are retained for audit. Historical features or cached paper scores cannot stand in for fresh computation.

The public API produces target-relative ranks. Compare each complete target to its historical final-score ordering; do not concatenate target-relative ranks and call the resulting pooled correlation a reproduction of the paper's global score. The reference is the frozen global332 stable ordinal rank of `rankscore_3view_base`, restricted to the exact35 observations only after ranking the full population.

Prospective engineering limits, locked before fresh candidate inference:

- Every observation/channel/seven-restart pair completes with finite values and supported chemistry preserved.
- Each target: absolute experimental Spearman-rho drift at most0.20 and historical score-order Spearman agreement at least0.70.
- Equal-target means: absolute rho drift at most0.10 and score-order agreement at least0.90, separately for all four targets and for the true disulfide pair. Both groups must pass.
- Undefined or constant cases are could-not-evaluate, never zero or PASS. Report all targets, uncertainty and outliers. No fitting, target removal or margin widening after results.

These limits describe a release smoke, not statistical equivalence. Official crystal inputs replace unavailable original predicted coordinates. A passing subset test does not establish exact historical-coordinate reproduction or full332/33 performance. **Fresh subset execution and independent result acceptance are pending; disulfide support remains an unqualified preview.** Release/platform parity separately exercises real orchestration and shared, hash-bound fresh feature tables.

## Historical checks

The older frozen released feature table contains **332 mutations/33 targets**. Its recorded PepDDG-ZS pooled rho is `0.6188807648016849`, with target-bootstrap interval `[0.5192060230325495,0.7078959993425558]`. The installed frozen benchmark command reproduces the score vector and these metrics:

```bash
python -m pepddg.frozen_benchmark --repo-root . --output /tmp/pepddg-frozen-replay
```

The command writes `scores.csv` and `report.json`, named `published_anchor_rank_regression_not_structure_to_score`. Experimental labels enter evaluation only. It validates the frozen rank recipe, without producing a fresh structural prediction.

The later rebuttal all-method common-coverage set has331/33 and recorded rho `0.6160850301058195`; another archive has340/34. These are distinct populations, not interchangeable labels or rank vectors. The platform historical raw-channel audit has reconstructed the released rank recipe separately. Original executed predicted-coordinate identities remain unavailable. Full332/33 and camera-ready cohort fresh runs are not required for this scoped release test and have not been performed.
