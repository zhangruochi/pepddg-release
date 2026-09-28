# Shipped benchmark data: provenance and rights

Only frozen evaluation tables, their metric/hash manifests and two structural
test fixtures are retained from the historical research snapshot. Development
experiments and intermediate data are maintained separately from this release.

| Retained files | Origin and purpose | Rights / attribution |
|---|---|---|
| `../results/v19_sanitized_baseline/main_eval_v19.csv` | 332-observation SKEMPI-derived frozen feature benchmark | SKEMPI v2.0; CC BY 4.0 |
| `../results/v19_sanitized_baseline/bpti_eval_v19.csv` and `ood_eval_v19.csv` | Historical BPTI/control evaluation tables | Derived from Heyne et al. 2021; CC BY 4.0 |
| `independent_validation/scoring_input/1CBW.pdb` and `3OTJ.pdb` | Prepared public-coordinate test fixtures | RCSB Protein Data Bank; CC0 |
| Metric and hash JSONs under `../results/` | Historical frozen evaluation records | Original MIT-covered snapshot; see `../../../LICENSE-MIT` |

Please cite the underlying sources alongside PepDDG:

- Jankauskaitė et al., *SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation*, Bioinformatics 35, 462–469 (2019), DOI: 10.1093/bioinformatics/bty635.
- Heyne et al., eLife 10:e64898 (2021), DOI: 10.7554/eLife.64898, for the historical BPTI-derived tables.
- RCSB Protein Data Bank for public structural coordinates.

The fresh four-target example has its own source identities, modifications and
SKEMPI attribution in [the example guide](../../../examples/skempi_cyclic/README.md).
Historical frozen features do not establish the original executed structure
lineage; see [benchmark limitations](../../../docs/BENCHMARK.md).

Data rights are separate from new code and documentation licensing. The original
MIT grant remains available for covered historical materials; new release
materials use PolyForm Noncommercial 1.0.0 as explained in
[license scope](../../../LICENSE_SCOPE.md).
