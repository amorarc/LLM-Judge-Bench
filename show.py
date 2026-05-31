#!/usr/bin/env python3
"""
Show per-phrase perplexity for every model in the config.

Usage
-----
    python show.py --samples 10
    python show.py --samples 20 --dataset yahoo --topic Health
    python show.py --samples 5  --dataset ethics --subset deontology
    python show.py --samples 10 --config configs/models.yaml --seed 7
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.huggingface_model import HuggingFaceModel
from loaders.ethics import SUBSETS as ETHICS_SUBSETS, sample_ethics
from loaders.yahoo import sample_yahoo_by_topic

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "huggingface_hub", "datasets", "transformers", "filelock", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("show")

PHRASE_WIDTH = 80
COL_W_MIN   = 12   # minimum column width (widens to fit model name)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-phrase perplexity viewer")
    p.add_argument("--samples", type=int, required=True,
                   help="Number of phrases to display")
    p.add_argument("--dataset", choices=["ethics", "yahoo"], default="ethics",
                   help="Dataset to sample from (default: ethics)")
    p.add_argument("--subset",  default="commonsense",
                   help="Ethics subset (commonsense|deontology|justice|utilitarianism|virtue)")
    p.add_argument("--topic",   default=None,
                   help="Yahoo topic (e.g. Health, Sports). Omit to pick the first available.")
    p.add_argument("--config",  default="configs/models.yaml")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _trunc(text: str, width: int = PHRASE_WIDTH) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def _load_texts(args: argparse.Namespace) -> tuple[list[str], str]:
    """Return (texts, label) for the chosen dataset."""
    if args.dataset == "ethics":
        subset = args.subset
        if subset not in ETHICS_SUBSETS:
            raise SystemExit(f"Unknown ethics subset {subset!r}. Choose from: {list(ETHICS_SUBSETS)}")
        texts = sample_ethics(subset, args.samples, seed=args.seed)
        return texts, f"ethics / {subset}"

    # yahoo
    all_topics = sample_yahoo_by_topic(args.samples, seed=args.seed)
    if args.topic:
        if args.topic not in all_topics:
            raise SystemExit(f"Unknown topic {args.topic!r}. Available: {list(all_topics)}")
        topic = args.topic
    else:
        topic = next(iter(all_topics))
    return all_topics[topic], f"yahoo / {topic}"


# ── Display ────────────────────────────────────────────────────────────────────

def print_table(
    texts: list[str],
    perplexities: np.ndarray,   # shape (n_models, n_texts)
    short_ids: list[str],
    vocab_sizes: list[int | None],
    label: str,
) -> None:
    col_w = max(COL_W_MIN, max(len(sid) for sid in short_ids))
    header_phrase = f"{'Phrase':<{PHRASE_WIDTH}}"
    header_models = "  ".join(f"{sid:>{col_w}}" for sid in short_ids)
    sep = "─" * (PHRASE_WIDTH + 2 + len(header_models))
    wide = "═" * len(sep)

    print(f"\n{wide}")
    print(f"  Dataset : {label}")
    print(f"  Phrases : {len(texts)}")
    print(f"{wide}")
    print(f"  {header_phrase}  {header_models}")
    print(f"  {sep}")

    for j, text in enumerate(texts):
        ppl_cols = "  ".join(f"{perplexities[i, j]:>{col_w}.2f}" for i in range(len(short_ids)))
        print(f"  {_trunc(text):<{PHRASE_WIDTH}}  {ppl_cols}")

    print(f"  {sep}")

    # mean perplexity
    means = perplexities.mean(axis=1)
    mean_cols = "  ".join(f"{m:>{col_w}.2f}" for m in means)
    print(f"  {'(mean ppl)':>{PHRASE_WIDTH}}  {mean_cols}")

    # vocabulary size
    vsz_cols = "  ".join(
        f"{v:>{col_w},d}" if v is not None else f"{'N/A':>{col_w}}"
        for v in vocab_sizes
    )
    print(f"  {'(vocab size)':>{PHRASE_WIDTH}}  {vsz_cols}")

    # vocab / mean entropy  (entropy = mean log-perplexity over the sample)
    mean_entropy = np.log(perplexities).mean(axis=1)   # mean ln(ppl) per model
    norm_cols = "  ".join(
        f"{vocab_sizes[i] / mean_entropy[i]:>{col_w}.2f}" if vocab_sizes[i] is not None
        else f"{'N/A':>{col_w}}"
        for i in range(len(short_ids))
    )
    print(f"  {'(vocab/entropy)':>{PHRASE_WIDTH}}  {norm_cols}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    models = load_models(args.config)
    short_ids = [_short(m.model_id) for m in models]

    texts, label = _load_texts(args)
    logger.info(f"Sampled {len(texts)} phrases from {label}")

    # n_models × n_texts perplexity matrix
    ppl = np.zeros((len(models), len(texts)))
    vocab_sizes: list[int | None] = []
    for i, model in enumerate(models):
        log_probs = model.batch_logprobs(texts)
        ppl[i] = np.exp(-np.array(log_probs))
        vocab_sizes.append(model.vocab_size)
        model.unload()

    print_table(texts, ppl, short_ids, vocab_sizes, label)


if __name__ == "__main__":
    main()
