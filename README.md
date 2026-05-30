# LLM Judge Diversity — Cross-Model Perplexity Matrices

Research-grade framework for evaluating the **diversity and redundancy of LLMs as text-quality judges** via a full cross-model score matrix.

## Core Idea

Define N models M₁…Mₙ and K texts x₁…xₖ. Compute the **score matrix** S:

```
S[i, j] = (1 / T_j) * Σ_t log p_{M_i}(x_{j,t} | x_{j,<t})
```

Then analyse: which models are redundant? Which provide unique signal? How stable are rankings across models?

## Quick Start

```bash
pip install -e ".[dev]"          # core + dev dependencies
pip install -e ".[huggingface]"  # add HuggingFace support
python examples/minimal_run.py   # full pipeline, no GPU needed
```

## Repository Structure

```
configs/              YAML config for models and benchmarks
datasets/             Dataset abstractions (JSON, CSV, HuggingFace)
models/               BaseModel + HuggingFace / OpenAI / vLLM wrappers
evaluation/           CrossEntropyMatrix with batching & disk caching
analysis/             Correlation, clustering, diversity, ranking
visualization/        Heatmap, PCA, dendrogram, similarity graph
benchmarks/           MMLU, GSM8K, HellaSwag loaders
pipelines/            CLI entry points for full evaluation
utils/                Logging, I/O, math helpers
tests/                pytest suite (no GPU required)
examples/             minimal_run.py + quickstart.ipynb
outputs/              Generated artefacts (gitignored)
```

## Pipelines

```bash
# Compute matrix from a text file (one sentence per line)
python -m pipelines.run_matrix \
    --models configs/models.yaml \
    --texts  my_corpus.txt \
    --output outputs/score_matrix.csv

# Run across multiple benchmarks
python -m pipelines.run_benchmarks \
    --models      configs/models.yaml \
    --benchmarks  configs/benchmarks.yaml \
    --output      outputs/

# End-to-end: matrix + analysis + all plots + markdown report
python -m pipelines.full_evaluation \
    --models  configs/models.yaml \
    --texts   my_corpus.txt \
    --output  outputs/ \
    --no-show
```

## Analysis Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Pearson r | corr(S[i,:], S[k,:]) | Pairwise model agreement |
| Diversity D(A) | mean(1 − corr_ik) | Ensemble informativeness |
| Kendall τ | kendall(rank(S[i,:]), rank(S[k,:])) | Ranking consistency |
| Disagreement | Var_i(S[:,j]) | Per-text model uncertainty |

## Outputs

| File | Description |
|------|-------------|
| `score_matrix.csv` | N×K raw log-prob scores |
| `model_correlations.json` | All pairwise Pearson r values |
| `diversity_report.json` | Diversity scores, optimal subsets |
| `ranking_stability.json` | Kendall τ summary |
| `diversity_report.md` | Human-readable summary |
| `plots/` | Heatmap, PCA, dendrogram, graph |

## Tests

```bash
pytest tests/ -v
```

All tests use synthetic mock models — no GPU or network required.

## Extending

**Add a new model backend**: subclass `BaseModel`, implement `logprobs` and optionally `batch_logprobs`, then register with `@ModelRegistry.register("mytype")`.

**Add a new benchmark**: subclass `BaseBenchmark`, implement `load() -> BaseDataset`.

**Add a new analysis**: operate directly on the `(N, K)` numpy score matrix S.
