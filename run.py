#!/usr/bin/env python3
"""
LLM Judge Diversity — main entry point.

Loads models from configs/models.yaml, samples X texts from each
hendrycks/ethics subset and each yahoo_answers_topics topic, computes a
perplexity matrix, and runs pairwise distribution tests (KS + Spearman)
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
from analysis.distribution_test import pairwise_tests, results_to_matrices, pca_opinion

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
    pca = pca_opinion(S, model_ids)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Perplexity matrix — full IDs in CSV for traceability
    pd.DataFrame(np.exp(-S).T, columns=model_ids).to_csv(out_dir / "perplexity.csv", index_label="text_idx")

    # Pairwise test CSVs — full IDs
    for key, mat in mats.items():
        pd.DataFrame(mat, index=model_ids, columns=model_ids).to_csv(out_dir / f"{key}.csv")

    # PCA — explained variance and loadings
    evr = pca["explained_variance_ratio"]
    pd.DataFrame({"pc": range(1, len(evr) + 1), "explained_variance_ratio": evr}).to_csv(
        out_dir / "pca_explained.csv", index=False
    )
    pd.DataFrame(pca["loadings"], columns=short_ids,
                 index=[f"PC{k+1}" for k in range(len(evr))]).to_csv(out_dir / "pca_loadings.csv")

    # Print table to stdout
    _print_table(results, name, short_ids, model_ids)

    # Plots
    _save_heatmap(mats, short_ids, name, out_dir / "heatmap.png")
    _save_pca_plot(evr, short_ids, pca["loadings"], name, out_dir / "pca_opinion.png")
    logger.info(f"[{name}] saved to {out_dir}")


def _print_table(results, name: str, short_ids: list[str], model_ids: list[str]) -> None:
    short = {mid: sid for mid, sid in zip(model_ids, short_ids)}
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Model A':30s}  {'Model B':30s}  {'KS stat':>8}  {'KS p':>8}  {'dCor':>8}  {'dCor p':>8}")
    for (a, b), res in results.items():
        sig = "***" if res.ks_pvalue < 0.01 else ("*" if res.ks_pvalue < 0.05 else "   ")
        print(f"  {short[a]:30s}  {short[b]:30s}  {res.ks_statistic:8.4f}  {res.ks_pvalue:8.4f}{sig}  {res.dcor_stat:8.4f}  {res.dcor_pvalue:8.4f}")


def _save_heatmap(mats: dict, short_ids: list[str], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(short_ids) * 2), max(4, len(short_ids) * 1.5)))

    ks_df = pd.DataFrame(mats["ks_statistic"], index=short_ids, columns=short_ids)
    dc_df = pd.DataFrame(mats["dcor_stat"],    index=short_ids, columns=short_ids)

    sns.heatmap(ks_df, ax=axes[0], annot=True, fmt=".3f", cmap="Reds",  vmin=0, vmax=1, square=True, linewidths=0.5)
    axes[0].set_title("KS statistic  (0=same dist, 1=different)")

    sns.heatmap(dc_df, ax=axes[1], annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, square=True, linewidths=0.5)
    axes[1].set_title("Distance Correlation  (0=independent, 1=dependent)")

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_pca_plot(evr: np.ndarray, short_ids: list[str], loadings: np.ndarray,
                   title: str, path: Path) -> None:
    n_pcs = len(evr)
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, n_pcs * 0.8 + 2)))

    # Left: explained variance bar chart
    axes[0].bar(range(1, n_pcs + 1), evr * 100, color="steelblue")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("PCA explained variance")
    axes[0].set_xticks(range(1, n_pcs + 1))

    # Right: PC1 loadings (model weights on the dominant opinion axis)
    axes[1].barh(short_ids, loadings[0], color="darkorange")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Loading on PC1")
    axes[1].set_title(f"PC1 loadings  ({evr[0]*100:.1f}% var)")

    fig.suptitle(title, fontsize=12)
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
