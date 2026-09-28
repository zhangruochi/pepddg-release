# Reproducibility and benchmark results

## Fresh structure-to-score examples

The community release includes 35 official SKEMPI v2.0 observations from the
four systems grouped as cyclic targets in the paper. The deposited 1SMF and
5XCO peptides have disulfide closures; 3EQS and 3EQY are linear PMI controls.
The reference run generated all three channels for every observation, using
seven paired WT/mutant restarts. 5XCO retains its ACE/NH2 terminal caps.

| Group | Mean absolute experimental-ρ drift | Mean paper-to-fresh score-order agreement |
|---|---:|---:|
| Four paper-defined targets | 0.0806 | 0.9019 |
| True disulfide pair: 1SMF and 5XCO | 0.0293 | 0.9250 |

All targets passed the predefined engineering limits: absolute correlation
drift ≤0.20 and score-order agreement ≥0.70. Both groups also passed the
aggregate limits: mean drift ≤0.10 and mean agreement ≥0.90.
These are small-cohort consistency criteria, not equivalence intervals.

After installing the environment, run from the repository root:

```bash
bash examples/skempi_cyclic/run.sh /path/to/output
```

The [example guide](../examples/skempi_cyclic/README.md) contains per-target
results, source attribution, prepared inputs, raw feature and score tables,
provenance and commands for generating PNG/SVG figures. Experimental labels
and archived scores are used only by the evaluator after inference.

These prepared official crystal structures omit receptor cofactors and waters.
They differ from the unavailable historical predicted inputs. The exercise
therefore tests broad rank consistency, rather than exact-input or bitwise
reproduction. It does not establish experimental affinity.

## Historical frozen benchmark

The frozen release table contains 332 mutations across 33 targets. Its recorded
PepDDG-ZS pooled Spearman correlation is `0.6188807648016849`, with a recorded
target-bootstrap 95% interval of `[0.5192060230325495, 0.7078959993425558]`.
The later rebuttal all-method common-coverage analysis contains 331 mutations
across 33 targets; those cohort identities must not be conflated.

Replay the frozen scoring and metric calculation:

```bash
python -m pepddg.frozen_benchmark --repo-root . --output /tmp/pepddg-frozen-replay
python -m pepddg.release_audit --repo-root . --json
```

The replay verifies source hashes, membership, the complete score vector,
pooled correlation and recorded bootstrap interval. The predictor receives
neither experimental labels nor recorded output scores. A successful replay
checks frozen-feature arithmetic; it is not fresh structural inference.

The manuscript describes the 332/33 headline as using crystal inputs, whereas
archived producer code and raw physics tables indicate predicted inputs.
The executed intermediates and original prepared structures are unavailable,
so this historical structure-provenance conflict remains unresolved. Do not
interpret the frozen table as a verified crystal-input benchmark. Full 332/33
fresh structural reproduction has not been completed; the fresh release
example covers the separate 35-observation subset above.

The frozen results remain at their original paths under
`research/pepddg_v5/results/` to preserve recorded hashes and source manifests.
That directory contains benchmark assets, not the ongoing research workspace.
See [data provenance](../research/pepddg_v5/data/DATA_PROVENANCE.md) and
[license scope](../LICENSE_SCOPE.md) for attribution and rights.
