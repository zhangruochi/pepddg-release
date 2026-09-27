# PepDDG 中文快速入门

PepDDG 用于比较**同一受体、同一母体肽的一组单点突变**。输出分数是相对排序：越低表示在本组中越优先；它不是实验亲和力，也不是以 kcal/mol 表示的结合自由能。

在 Linux x86_64 上，从本仓库创建完整 CPU 环境：

```bash
conda env create -f environment.yaml
conda activate pepddg
pepddg doctor --json
```

`doctor` 仅检查依赖与随包模型文件是否存在，不运行结构计算。若只需给已有的六列原始特征打分，可以在现有 Python 3.12 环境中运行 `python -m pip install .`。

先试已有特征表：

```bash
pepddg score-features --input examples/raw_features.csv --output /tmp/pepddg-scores.csv
```

为新的**线性肽**从结构生成特征并排序，需要一个含受体链和肽链的 PDB/mmCIF 文件，以及如下突变 CSV：

```csv
mutation,chain,resnum,icode,wt,mut
TI11A,I,11,,T,A
```

运行时把链 ID、残基编号和野生型氨基酸替换成实际结构中的值：

```bash
pepddg score-structures \
  --structure /path/to/complex.pdb \
  --peptide-chain I --receptor-chain F \
  --mutations /path/to/mutations.csv \
  --target target_001 --parent-id WT \
  --output /tmp/pepddg-structural-result
```

默认对 WT 和每个突变体分别进行七次配对 OpenMM 计算，在 CPU 上可能很慢。成功后读取 `scores.csv`，同时保留 `features.csv`、`provenance.json`、输入结构和突变表。重复使用相同的输出目录时，工具只接受匹配的中断检查点。当前结构路径只支持标准氨基酸组成的线性肽和一条受体链；环肽、二硫键闭环、修饰残基等需要另行验证的拓扑流程。

完整的输入约束见 [结构工作流](STRUCTURES.md)，论文 SKEMPI 结果与当前验证边界见 [基准说明](BENCHMARK.md)。新发布材料允许非商业研究使用；公司内部研发等商业用途需单独书面授权，详见 [许可范围](../LICENSE_SCOPE.md)。
