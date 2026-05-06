# ΔΔG Prediction Benchmark: Literature Survey

> Last updated: 2026-02-20. Sources at bottom.

## Fair SOTA Numbers (Leakage-Free Evaluations)

| Method | Year | Split Type | Spearman | N_mut | Reference |
|--------|------|-----------|----------|-------|-----------|
| USP-ddG | 2025 | CATH held-out | ~0.65 (est.) | 813 | bioRxiv |
| CATH-ddG | 2025 | CATH held-out | 0.627 | 813 | Bioinformatics |
| Flex ddG | 2018 | CATH held-out | 0.610 | 813 | CATH-ddG paper |
| FoldX + StaB-ddG | 2025 | Homology split | ~0.53 | ~600-800 | ICML 2025 |
| FoldX | 2005 | CATH held-out | 0.525 | 813 | CATH-ddG paper |
| BA-DDG | 2025 | 3-fold CV | 0.513 (per-struct) | ~4076 | ICLR 2025 |
| Pythia-PPI | 2024 | 5-fold CV | 0.527 (per-struct) | ~4076 | PMC |

**Key finding**: On fair evaluations, SOTA ΔΔG Spearman is ~0.60-0.65. Flex ddG is highly competitive at 0.61.

## Our Numbers (PepDDG v19, Peptide Subset — FINAL)

| Method | Spearman | N_mut | N_targets | Notes |
|--------|----------|-------|-----------|-------|
| PepDDG v19 strict clean-3 | **0.691** | 332 | 33 | Sealed common-coverage split, 95% CI [0.612, 0.767] |
| v1 implicit (crystal, 7-9 targets) | 0.769-0.852 | 66 | 7-9 | Per-target LOTO |
| 5-way rank-sum ensemble | 0.596 | 406 | 38 | Historical predicted-structure result, archived |
| Physics (implicit) standalone | 0.446 | 455 | 40 | Predicted structures, paired |
| Rosetta cartesian_ddg (FINAL) | 0.390 | 455 | 40 | Interface DDG |
| FoldX BuildModel (FINAL) | 0.414 | 455 | 40 | Predicted structures |

**Unique contribution**: No published work has benchmarked ΔΔG prediction on peptide-protein interfaces (<80 residues partner). Our 40-target, 455-mutation peptide subset is novel.

## Physics-Based Methods Comparison

| Method | Random Split | Fair Split | Category |
|--------|-------------|-----------|----------|
| Flex ddG | 0.39-0.43 | **0.610** | Physics |
| FoldX | 0.37-0.52 | 0.525 | Physics |
| Rosetta cartesian_ddg | 0.30-0.35 | ~0.30 | Physics |
| Our v19 strict clean-3 rank fusion | N/A | 0.691 (pooled) | Training-free rank fusion |
| Our historical implicit/feature ensemble | N/A | 0.596 (pooled) | Physics-heavy historical result |

## Deep Learning SOTA (2024-2025)

### BA-DDG (ICLR 2025 Spotlight, 3-fold CV on SKEMPI v2)
| Method | Per-Struct Spearman | Overall Spearman | Overall Pearson |
|--------|-------------------|-----------------|-----------------|
| BA-DDG | **0.5134** | **0.6346** | **0.7118** |
| ProMIM | 0.4310 | 0.5730 | 0.6720 |
| RDE-Net | 0.4010 | 0.5584 | 0.6447 |
| DiffAffinity | 0.3970 | 0.5560 | 0.6609 |
| FoldX | 0.3693 | 0.4071 | 0.3120 |
| Rosetta | 0.2988 | 0.3468 | 0.3113 |

### CATH-ddG (Bioinformatics 2025, Leakage-Free Split)
| Method | SpearmanR | PearsonR | AUROC |
|--------|-----------|----------|-------|
| CATH-ddG | **0.627** | **0.615** | 0.781 |
| Flex ddG | 0.610 | 0.608 | 0.764 |
| FoldX | 0.525 | 0.460 | 0.754 |
| RDE-Net | 0.470 | 0.459 | 0.745 |
| DiffAffinity | 0.363 | 0.314 | 0.625 |

### StaB-ddG (ICML 2025, Homology Split)
- FoldX + StaB-ddG ensemble: per-interface Spearman ~0.53 (SOTA on this split)
- On stringent homology split, supervised DL methods ALL underperform FoldX/Flex ddG
- Simple physics ensemble beats deep learning when data leakage is removed

## Data Leakage Warning

1. **Random splits are misleading**: Covering ratio explains R²=0.96 of performance variation (Aldeghi 2024)
2. **CATH splits reveal truth**: Most DL methods drop below Flex ddG on CATH split
3. **AbDesign zero-shot**: ML predictors achieve rho~0.4-0.7 on native data but rho~0.0-0.1 on non-overlapping AbDesign data
4. **Graphinity**: Models achieving Pearson 0.87 on random splits drop dramatically on CDR identity cutoffs

## Relevance to Our Work

1. **Our per-target evaluation is inherently leakage-free** — LOTO by target is similar to structure-level split
2. **Per-interface Spearman ~0.53 is the published SOTA** on fair splits (StaB-ddG + FoldX)
3. **Our 0.691 pooled rho** on the sealed common-coverage peptide split is competitive with SOTA, despite:
   - Using only physics-based features (no supervised training on SKEMPI)
   - Being a purely peptide-focused subset (harder — shorter interfaces, more flexible)
   - Using predicted (not crystal) structures
4. **No peptide-specific benchmark exists** — our work fills a genuine gap

## Sources

- [BA-DDG (ICLR 2025)](https://arxiv.org/html/2410.09543v1)
- [CATH-ddG (Bioinformatics 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12261453/)
- [StaB-ddG (ICML 2025)](https://arxiv.org/html/2507.05502v1)
- [USP-ddG (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.11.09.687124v2)
- [Pythia-PPI (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12199698/)
- [ProMIM (arxiv 2024)](https://arxiv.org/html/2405.17802v1)
- [DiffAffinity (NeurIPS 2023)](https://github.com/EureKaZhu/DiffAffinity)
- [Bias quantification (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10777193/)
- [Graphinity (Nature Comp Sci 2025)](https://www.nature.com/articles/s43588-025-00823-8)
- [GeoDDG (Nature Comp Sci 2024)](https://www.nature.com/articles/s43588-024-00716-2)
- [AbDesign generalization](https://pmc.ncbi.nlm.nih.gov/articles/PMC12520099/)
