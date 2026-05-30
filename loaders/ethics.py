import random
from typing import Optional

# hendrycks/ethics subsets and the column that holds the main text
SUBSETS = {
    "commonsense":    "input",
    "deontology":     "scenario",
    "justice":        "scenario",
    "utilitarianism": "baseline",
    "virtue":         "scenario",
}


def sample_ethics(subset: str, n: int, seed: int = 42, split: str = "test") -> list[str]:
    """
    Sample n texts from a hendrycks/ethics subset.

    Parameters
    ----------
    subset : one of SUBSETS keys
    n      : number of texts to sample (capped at dataset size)
    seed   : random seed for reproducibility
    split  : dataset split ("train" or "test")
    """
    from datasets import load_dataset

    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}. Choose from: {list(SUBSETS)}")

    path = f"hf://datasets/hendrycks/ethics/data/{subset}/{split}.csv"
    ds = load_dataset("csv", data_files={split: path}, split=split)
    field = SUBSETS[subset]
    texts = [str(row[field]) for row in ds if row.get(field)]
    rng = random.Random(seed)
    return rng.sample(texts, min(n, len(texts)))
