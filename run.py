#!/usr/bin/env python3
"""
LLM Judge Diversity — main entry point.

Loads models from configs/models.yaml, samples X texts from each
hendrycks/ethics subset and each yahoo_answers_topics topic, computes a
perplexity matrix, and runs pairwise Spearman + distance-correlation tests
between every pair of models.

Usage
-----
    python run.py --n-samples 100
    python run.py --n-samples 50 --output results/ --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.huggingface_model import HuggingFaceModel
from loaders.ethics import SUBSETS as ETHICS_SUBSETS, sample_ethics
from loaders.yahoo import sample_yahoo_by_topic
from evaluation.matrix import compute_matrix
from analysis.distribution_test import pairwise_tests, results_to_matrices

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "huggingface_hub", "datasets", "transformers", "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("run")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Judge Diversity")
    p.add_argument("--n-samples", type=int, required=True,
                   help="Number of texts to sample per dataset / subset / topic")
    p.add_argument("--config",  default="configs/models.yaml")
    p.add_argument("--output",  default="outputs/")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────

def load_models(config_path: str) -> list[HuggingFaceModel]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    models = []
    for entry in cfg["models"]:
        if isinstance(entry, str):
            entry = {"id": entry}
        mid = entry.pop("id")
        models.append(HuggingFaceModel(mid, **entry))
    logger.info(f"Models: {[m.model_id for m in models]}")
    return models


# ── Analysis + saving ──────────────────────────────────────────────────────

def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def run_and_save(
    models: list[HuggingFaceModel],
    texts: list[str],
    name: str,
    out_dir: Path,
) -> None:
    """Compute matrix → run tests → save CSVs and heatmap for one corpus."""
    model_ids = [m.model_id for m in models]
    short_ids = [_short(mid) for mid in model_ids]
    logger.info(f"[{name}] computing matrix ({len(model_ids)} models × {len(texts)} texts)")

    S = compute_matrix(models, texts)
    results = pairwise_tests(S, model_ids)
    mats = results_to_matrices(results, model_ids)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Perplexity matrix — full IDs in CSV for traceability
    pd.DataFrame(np.exp(-S).T, columns=model_ids).to_csv(out_dir / "perplexity.csv", index_label="text_idx")

    # Pairwise test CSVs — full IDs
    for key, mat in mats.items():
        pd.DataFrame(mat, index=model_ids, columns=model_ids).to_csv(out_dir / f"{key}.csv")

    _print_table(results, name, short_ids, model_ids)
    _save_heatmap(mats, short_ids, name, out_dir / "heatmap.png")
    logger.info(f"[{name}] saved to {out_dir}")


def _print_table(results, name: str, short_ids: list[str], model_ids: list[str]) -> None:
    short = {mid: sid for mid, sid in zip(model_ids, short_ids)}
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Model A':30s}  {'Model B':30s}  {'Spearman r':>10}  {'Spearman p':>10}  {'dCor':>8}  {'dCor p':>8}")
    for (a, b), res in results.items():
        sig = "***" if res.spearman_pvalue < 0.01 else ("*" if res.spearman_pvalue < 0.05 else "   ")
        print(f"  {short[a]:30s}  {short[b]:30s}  {res.spearman_r:10.4f}  {res.spearman_pvalue:10.4f}{sig}  {res.dcor_stat:8.4f}  {res.dcor_pvalue:8.4f}")


def _save_heatmap(mats: dict, short_ids: list[str], title: str, path: Path) -> None:
    n = len(short_ids)
    fig, axes = plt.subplots(1, 2, figsize=(max(16, n * 3), max(4, n * 1.5)))

    sp_df = pd.DataFrame(mats["spearman_r"], index=short_ids, columns=short_ids)
    dc_df = pd.DataFrame(mats["dcor_stat"],  index=short_ids, columns=short_ids)

    sns.heatmap(sp_df, ax=axes[0], annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1, square=True, linewidths=0.5)
    axes[0].set_title("Spearman r  (1=same ranking, −1=opposite)")

    sns.heatmap(dc_df, ax=axes[1], annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, square=True, linewidths=0.5)
    axes[1].set_title("Distance Correlation  (0=independent, 1=dependent)")

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out = Path(args.output)
    models = load_models(args.config)

    # ── hendrycks/ethics ──────────────────────────────────────────────
    logger.info("=== Dataset: hendrycks/ethics ===")
    for subset in ETHICS_SUBSETS:
        texts = sample_ethics(subset, args.n_samples, seed=args.seed)
        logger.info(f"  subset={subset}, sampled {len(texts)} texts")
        run_and_save(models, texts, f"ethics/{subset}", out / "ethics" / subset)

    # ── yahoo_answers_topics ──────────────────────────────────────────
    logger.info("=== Dataset: yahoo_answers_topics ===")
    topics = sample_yahoo_by_topic(args.n_samples, seed=args.seed)
    for topic, texts in topics.items():
        logger.info(f"  topic={topic}, sampled {len(texts)} texts")
        run_and_save(models, texts, f"yahoo/{topic}", out / "yahoo" / topic)

    logger.info(f"\nDone. All outputs in: {out.resolve()}")


if __name__ == "__main__":
    main()
