from __future__ import annotations

import numpy as np
import dcor
from scipy import stats
from dataclasses import dataclass


@dataclass
class PairResult:
    ks_statistic: float   # 0 = identical distributions, 1 = maximally different
    ks_pvalue: float      # small → reject H0 (same distribution)
    dcor_stat: float      # distance correlation (0 = independent, 1 = dependent)
    dcor_pvalue: float


def pairwise_tests(S: np.ndarray, model_ids: list[str]) -> dict[tuple[str, str], PairResult]:
    """
    For each ordered pair (M_i, M_j), test whether their perplexity distributions
    on the same K texts are drawn from the same population.

    Uses:
      - 2-sample KS test   (H0: same continuous distribution)
      - Spearman correlation (ranking agreement of the perplexity vectors)

    Parameters
    ----------
    S : (N, K) array of mean log-probs per token
    model_ids : length-N list of model names

    Returns
    -------
    dict keyed by (model_i, model_j) for all i < j
    """
    # Convert log-probs to perplexity: ppl = exp(-mean_logprob)
    ppl = np.exp(-S)
    N = len(model_ids)
    results: dict[tuple[str, str], PairResult] = {}

    for i in range(N):
        for j in range(i + 1, N):
            ks = stats.ks_2samp(ppl[i], ppl[j])
            dc_coef = dcor.distance_correlation(ppl[i], ppl[j])
            dc_test = dcor.independence.distance_correlation_t_test(ppl[i], ppl[j])
            results[(model_ids[i], model_ids[j])] = PairResult(
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
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
    Diagonal is set to semantically appropriate defaults (0 for KS, 1 for Spearman).
    """
    N = len(model_ids)
    idx = {m: i for i, m in enumerate(model_ids)}

    ks_stat   = np.zeros((N, N))          # diagonal = 0: no difference with itself
    ks_pval   = np.ones((N, N))           # diagonal = 1: can't reject same distribution
    dc_stat   = np.eye(N)                 # diagonal = 1: perfect correlation with itself
    dc_pval   = np.zeros((N, N))          # diagonal = 0: self-correlation is certain

    for (a, b), res in results.items():
        i, j = idx[a], idx[b]
        ks_stat[i, j]  = ks_stat[j, i]  = res.ks_statistic
        ks_pval[i, j]  = ks_pval[j, i]  = res.ks_pvalue
        dc_stat[i, j]  = dc_stat[j, i]  = res.dcor_stat
        dc_pval[i, j]  = dc_pval[j, i]  = res.dcor_pvalue

    return {"ks_statistic": ks_stat, "ks_pvalue": ks_pval,
            "dcor_stat": dc_stat, "dcor_pvalue": dc_pval}


def pca_opinion(S: np.ndarray, model_ids: list[str]) -> dict:
    """
    PCA on the (K_texts × N_models) perplexity matrix to test whether models
    share a dominant 'hardness' opinion.

    If PC1 explained variance is high (e.g. > 0.8 with N=2 models, or dominates
    otherwise), a single latent opinion axis captures most of the variance.

    Returns
    -------
    explained  : (N,) array of explained-variance ratios, one per PC
    loadings   : (N, N_models) array — loadings[k] are model weights on PC k
    """
    ppl = np.exp(-S).T          # (K_texts, N_models)
    ppl = ppl - ppl.mean(axis=0)  # center per model
    _, s, Vt = np.linalg.svd(ppl, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    return {
        "explained_variance_ratio": explained,
        "loadings": Vt,          # row k = PC-k direction in model space
        "model_ids": model_ids,
    }
