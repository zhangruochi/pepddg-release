# Changelog

## 0.1.1

- Add `pepddg.predict(wt_structure, mutations, ...)`, returning a DataFrame directly for one mutation or an ordered mutation series.
- Recognize validated linear/disulfide topology without requiring mutation manifests or input objects.
- Keep raw channels for single candidates and mark their relative ranks unavailable; preserve the multi-candidate scoring recipe.
- Retain automatic output/checkpoint directories and expose provenance in DataFrame metadata.

## 0.1.0

- Installable `pepddg` Python package and command-line interface with a pinned conda environment.
- Strict raw-feature scoring and complete three-channel structure-to-score inference.
- Validated disulfide integrity and the exact ACE/NH2 terminal caps used by the included 5XCO example.
- Seven paired OpenMM restarts, bundled ProteinMPNN scoring, input checks and exact-input checkpoints.
- One-command reproduction of 35 observations from four paper-defined SKEMPI targets, including two true disulfide systems and two linear controls.
- Raw reference outputs, per-target metrics, experimental/ranking scatter plots and paper-to-fresh comparisons in PNG and SVG.
- Frozen-feature benchmark replay, scientific limitations, noncommercial terms and preserved historical/third-party rights.

See the [benchmark guide](docs/BENCHMARK.md) for the distinct scopes of fresh
subset inference and historical frozen-feature replay.
