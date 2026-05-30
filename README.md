# LLM Judge Diversity — Cross-Model Perplexity Analysis

Framework for evaluating the **diversity and redundancy of LLMs as text-quality judges** via cross-model perplexity matrices and pairwise distribution tests.

## Core Idea

Define N models M₁…Mₙ and K texts x₁…xₖ. Compute the **score matrix** S:

```
S[i, j] = (1 / T_j) * Σ_t log p_{M_i}(x_{j,t} | x_{j,<t})
```

Then test: do different models assign statistically similar perplexity distributions? Which models are redundant? Where do they disagree?

## Quick Start

```bash
pip install -e .
python run.py --n-samples 100
```

## Repository Structure

```
configs/          YAML config for models
loaders/          Dataset loaders (hendrycks/ethics, yahoo_answers_topics)
models/           BaseModel + HuggingFace causal-LM wrapper
evaluation/       Cross-model perplexity matrix computation
analysis/         Pairwise KS tests, distance correlation, PCA
outputs/          Generated artefacts (gitignored)
```

## Usage

```bash
python run.py --n-samples 100
python run.py --n-samples 50 --output results/ --seed 0 --config configs/models.yaml
```

## Datasets

| Dataset | Source | Subsets / Topics |
|---------|--------|-----------------|
| `hendrycks/ethics` | HuggingFace | commonsense, deontology, justice, utilitarianism, virtue |
| `yahoo_answers_topics` | HuggingFace | 10 topic categories (Society, Science, Health, …) |

## Analysis

For each corpus (ethics subset or yahoo topic):

| Metric | Description |
|--------|-------------|
| Spearman r | Ranking agreement of perplexity vectors (1 = same ranking, −1 = opposite) |
| Spearman p-value | Significance of the correlation |
| Distance correlation | dcor statistic (0 = independent, 1 = fully dependent) |
| dcor p-value | Significance of the distance correlation |

## Outputs

Per corpus (e.g. `outputs/ethics/commonsense/`):

| File | Description |
|------|-------------|
| `perplexity.csv` | K×N perplexity scores (rows = texts, cols = models) |
| `spearman_r.csv` | N×N Spearman r matrix |
| `spearman_pvalue.csv` | N×N Spearman p-value matrix |
| `dcor_stat.csv` | N×N distance correlation matrix |
| `dcor_pvalue.csv` | N×N distance correlation p-value matrix |
| `heatmap.png` | Spearman r + dcor heatmaps side by side |

## Adding Models

Edit `configs/models.yaml`. Each entry can be a bare model ID or a dict:

```yaml
models:
  - id: meta-llama/Llama-3.2-3B
    dtype: float16
    batch_size: 1
    quantization: int4   # int4 | int8 | omit for no quantization
```

Supported parameters: `dtype`, `device`, `batch_size`, `max_length`, `num_threads`, `quantization`.

## Extending

**Add a model backend**: subclass `BaseModel` in `models/base.py`, implement `logprobs` and optionally `batch_logprobs`.

**Add a dataset**: add a loader function in `loaders/` that returns `list[str]`, then call it from `run.py`.

**Add an analysis**: operate on the `(N, K)` numpy score matrix S returned by `evaluation.matrix.compute_matrix`.
