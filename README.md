# LLM Judge Diversity — Cross-Model Perplexity Analysis

Framework for evaluating the **diversity and redundancy of LLMs as text-quality judges** via cross-model perplexity matrices and pairwise distribution tests.

## Core Idea

Define $N$ models $M_1, \ldots, M_N$ and $K$ texts $x_1, \ldots, x_K$. Compute the **score matrix** $S \in \mathbb{R}^{N \times K}$:

$$S[i,\, j] = \frac{1}{T_j} \sum_{t=1}^{T_j} \log p_{M_i}\!\left(x_{j,t} \mid x_{j,<t}\right)$$

where $T_j$ is the token length of text $x_j$. The corresponding **perplexity** is:

$$\mathrm{PPL}_{M_i}(x_j) = \exp\!\left(-S[i,\, j]\right)$$

Then test: do different models assign statistically similar perplexity distributions? Which models are redundant? Where do they disagree?

## Quick Start

```bash
pip install -r requirements.txt
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

### Full evaluation (`run.py`)

Computes the full perplexity matrix, pairwise statistics, and heatmaps for all datasets:

```bash
python run.py --n-samples 100
python run.py --n-samples 50 --output results/ --seed 0 --config configs/models.yaml
```

### Per-phrase viewer (`show.py`)

Prints each phrase alongside the perplexity every model assigns to it:

```bash
# ethics subsets (default: commonsense)
python show.py --samples 10
python show.py --samples 5 --subset deontology

# yahoo topics
python show.py --samples 20 --dataset yahoo --topic Health
python show.py --samples 10 --dataset yahoo --topic Sports
```

Example output:

```
══════════════════════════════════════════════════════════════════════════════════════════════
  Dataset : ethics / commonsense
  Phrases : 5
══════════════════════════════════════════════════════════════════════════════════════════════
  Phrase                                                                          gemma-4-E2B  Qwen3.5-4B  Llama-3.2-3B
  ──────────────────────────────────────────────────────────────────────────────────────────
  He helped the old lady cross the street safely.                                      12.34       15.21         18.90
  She returned the wallet she found on the ground.                                      9.87       11.02         14.55
  …
  ──────────────────────────────────────────────────────────────────────────────────────────
                                                                          (mean)        11.10       13.11         16.72
```

## Datasets

| Dataset | Source | Subsets / Topics |
|---------|--------|-----------------|
| `hendrycks/ethics` | HuggingFace | commonsense, deontology, justice, utilitarianism, virtue |
| `yahoo_answers_topics` | HuggingFace | 10 topic categories (Society, Science, Health, …) |

## Analysis

For each pair of models $(M_i, M_j)$, two agreement metrics are computed over their $K$-dimensional perplexity vectors $\mathbf{s}_i, \mathbf{s}_j \in \mathbb{R}^K$:

### Spearman Rank Correlation

$$r_s(M_i, M_j) = 1 - \frac{6\displaystyle\sum_{k=1}^{K} d_k^2}{K(K^2 - 1)}$$

where $d_k = \mathrm{rank}(s_{i,k}) - \mathrm{rank}(s_{j,k})$ is the rank difference for text $k$.  
$r_s = 1$ means identical ranking; $r_s = -1$ means fully reversed; $r_s \approx 0$ means no agreement.

### Distance Correlation

$$\mathrm{dCor}(M_i, M_j) = \sqrt{\frac{\mathrm{dCov}^2(\mathbf{s}_i,\, \mathbf{s}_j)}{\sqrt{\mathrm{dCov}^2(\mathbf{s}_i,\, \mathbf{s}_i)\cdot \mathrm{dCov}^2(\mathbf{s}_j,\, \mathbf{s}_j)}}}$$

Unlike Spearman, $\mathrm{dCor} = 0$ implies statistical independence (not just linear/monotone independence). $\mathrm{dCor} = 1$ means fully dependent.

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Spearman $r_s$ | $[-1,\, 1]$ | Ranking agreement |
| Spearman $p$-value | $[0,\, 1]$ | Significance of $r_s$ |
| $\mathrm{dCor}$ | $[0,\, 1]$ | General dependence |
| $\mathrm{dCor}$ $p$-value | $[0,\, 1]$ | Significance of $\mathrm{dCor}$ |

## Outputs

Per corpus (e.g. `outputs/ethics/commonsense/`):

| File | Description |
|------|-------------|
| `perplexity.csv` | $K \times N$ perplexity scores (rows = texts, cols = models) |
| `spearman_r.csv` | $N \times N$ Spearman $r_s$ matrix |
| `spearman_pvalue.csv` | $N \times N$ Spearman $p$-value matrix |
| `dcor_stat.csv` | $N \times N$ distance correlation matrix |
| `dcor_pvalue.csv` | $N \times N$ distance correlation $p$-value matrix |
| `heatmap.png` | Spearman $r_s$ + $\mathrm{dCor}$ heatmaps side by side |

## Results Gallery

> Images are generated locally under `outputs/` (gitignored). Run the pipeline first to populate them.

### Ethics — Heatmaps

<table>
  <tr>
    <th>Subset</th>
    <th>Heatmap (Spearman $r_s$ / dCor)</th>
  </tr>
  <tr>
    <td><b>Commonsense</b></td>
    <td><img src="outputs/ethics/commonsense/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Deontology</b></td>
    <td><img src="outputs/ethics/deontology/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Justice</b></td>
    <td><img src="outputs/ethics/justice/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Utilitarianism</b></td>
    <td><img src="outputs/ethics/utilitarianism/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Virtue</b></td>
    <td><img src="outputs/ethics/virtue/heatmap.png" width="100%"/></td>
  </tr>
</table>

---

### Yahoo Answers Topics — Heatmaps

<table>
  <tr>
    <th>Topic</th>
    <th>Heatmap (Spearman $r_s$ / dCor)</th>
  </tr>
  <tr>
    <td><b>Business & Finance</b></td>
    <td><img src="outputs/yahoo/Business_Finance/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Computers & Internet</b></td>
    <td><img src="outputs/yahoo/Computers_Internet/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Education & Reference</b></td>
    <td><img src="outputs/yahoo/Education_Reference/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Entertainment & Music</b></td>
    <td><img src="outputs/yahoo/Entertainment_Music/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Family & Relationships</b></td>
    <td><img src="outputs/yahoo/Family_Relationships/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Health</b></td>
    <td><img src="outputs/yahoo/Health/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Politics & Government</b></td>
    <td><img src="outputs/yahoo/Politics_Government/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Science & Mathematics</b></td>
    <td><img src="outputs/yahoo/Science_Mathematics/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Society & Culture</b></td>
    <td><img src="outputs/yahoo/Society_Culture/heatmap.png" width="100%"/></td>
  </tr>
  <tr>
    <td><b>Sports</b></td>
    <td><img src="outputs/yahoo/Sports/heatmap.png" width="100%"/></td>
  </tr>
</table>

---

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

**Add an analysis**: operate on the $(N \times K)$ numpy score matrix $S$ returned by `evaluation.matrix.compute_matrix`.
