import numpy as np
from models.base import BaseModel


def compute_matrix(models: list[BaseModel], texts: list[str]) -> np.ndarray:
    """
    Compute the (N, K) score matrix.

    S[i, j] = mean log-prob per token of text j under model i.
    Models are loaded and unloaded one at a time to avoid OOM.
    """
    rows = []
    for m in models:
        rows.append(np.array(m.batch_logprobs(texts), dtype=float))
        if hasattr(m, "unload"):
            m.unload()
    return np.stack(rows)
