from __future__ import annotations

import numpy as np
import dcor
from scipy import stats
from dataclasses import dataclass


@dataclass
class PairResult:
    spearman_r: float      # 1 = perfect ranking agreement, -1 = perfect disagreement
    spearman_pvalue: float # small → significant correlation
    dcor_stat: float       # distance correlation (0 = independent, 1 = dependent)
    dcor_pvalue: float


def pairwise_tests(S: np.ndarray, model_ids: list[str]) -> dict[tuple[str, str], PairResult]:
    """
    For each ordered pair (M_i, M_j), test whether their perplexity distributions
    on the same K texts are drawn from the same population.

    Uses:
      - Spearman correlation (ranking agreement of the perplexity vectors)
      - Distance correlation  (nonlinear dependence)

    Parameters
    ----------
    S : (N, K) array of mean log-probs per token
    model_ids : length-N list of model names

    Returns
    -------
    dict keyed by (model_i, model_j) for all i < j
    """
    ppl = np.exp(-S)
    N = len(model_ids)
    results: dict[tuple[str, str], PairResult] = {}

    for i in range(N):
        for j in range(i + 1, N):
            sp = stats.spearmanr(ppl[i], ppl[j])
            dc_coef = dcor.distance_correlation(ppl[i], ppl[j])
            dc_test = dcor.independence.distance_correlation_t_test(ppl[i], ppl[j])
            results[(model_ids[i], model_ids[j])] = PairResult(
                spearman_r=float(sp.statistic),
                spearman_pvalue=float(sp.pvalue),
                dcor_stat=float(dc_coef),
                dcor_pvalue=float(dc_test.pvalue),
            )

    return results


def results_to_matrices(
    results: dict[tuple[str, str], PairResult],
    model_ids: list[str],
) -> dict[str, np.ndarray]:
    """
    Unpack pairwise results into symmetric N×N numpy arrays.
    Diagonal is set to semantically appropriate defaults (1 for Spearman/dcor).
    """
    N = len(model_ids)
    idx = {m: i for i, m in enumerate(model_ids)}

    sp_r    = np.eye(N)     # diagonal = 1: perfect correlation with itself
    sp_pval = np.zeros((N, N))  # diagonal = 0: self-correlation is certain
    dc_stat = np.eye(N)     # diagonal = 1: perfect correlation with itself
    dc_pval = np.zeros((N, N))  # diagonal = 0: self-correlation is certain

    for (a, b), res in results.items():
        i, j = idx[a], idx[b]
        sp_r[i, j]    = sp_r[j, i]    = res.spearman_r
        sp_pval[i, j] = sp_pval[j, i] = res.spearman_pvalue
        dc_stat[i, j] = dc_stat[j, i] = res.dcor_stat
        dc_pval[i, j] = dc_pval[j, i] = res.dcor_pvalue

    return {"spearman_r": sp_r, "spearman_pvalue": sp_pval,
            "dcor_stat": dc_stat, "dcor_pvalue": dc_pval}
