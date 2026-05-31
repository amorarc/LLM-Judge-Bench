#!/usr/bin/env python3
"""
Cluster LLM models by their pairwise perplexity correlation.

Combines spearman_r (weight 0.2) and dcor_stat (weight 0.8), averaged across
all dataset subsets, then produces:
  • A network graph where edge thickness/color encodes correlation strength
    and node colors encode hierarchical cluster membership
  • A reordered correlation heatmap

Usage
-----
    python cluster_models.py
    python cluster_models.py --dataset yahoo
    python cluster_models.py --metric dcor_stat
    python cluster_models.py --outputs outputs/ --save cluster.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent

SPEARMAN_W = 0.2
DCOR_W     = 0.8


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster LLM models by perplexity correlation")
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--dataset", choices=["yahoo", "ethics", "all"], default="all")
    p.add_argument("--metric", choices=["spearman_r", "dcor_stat", "both"], default="both",
                   help="both uses 0.2×spearman_r + 0.8×dcor_stat (default)")
    p.add_argument("--method", default="average",
                   choices=["average", "ward", "complete", "single"])
    p.add_argument("--save", default=None)
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_matrices(outputs_dir: Path, dataset: str, metric: str) -> list[pd.DataFrame]:
    pattern = f"**/{metric}.csv" if dataset == "all" else f"{dataset}/**/{metric}.csv"
    result = []
    for path in sorted(outputs_dir.glob(pattern)):
        result.append(pd.read_csv(path, index_col=0))
    return result


def weighted_avg_corr(outputs_dir: Path, dataset: str, metric: str) -> tuple[pd.DataFrame, str, int]:
    """Return (avg_corr_df, metric_label, n_matrices)."""
    if metric == "both":
        s_mats = load_matrices(outputs_dir, dataset, "spearman_r")
        d_mats = load_matrices(outputs_dir, dataset, "dcor_stat")
        if not s_mats or not d_mats:
            sys.exit("Could not find both spearman_r and dcor_stat CSVs.")
        models = s_mats[0].index.tolist()

        def mean_stack(mats):
            return np.stack([m.loc[models, models].values for m in mats]).mean(axis=0)

        avg = SPEARMAN_W * mean_stack(s_mats) + DCOR_W * mean_stack(d_mats)
        label = f"0.2×spearman_r + 0.8×dcor_stat"
        n = len(s_mats)
    else:
        mats = load_matrices(outputs_dir, dataset, metric)
        if not mats:
            sys.exit(f"No '{metric}.csv' files found.")
        models = mats[0].index.tolist()
        avg = np.stack([m.loc[models, models].values for m in mats]).mean(axis=0)
        label = metric.replace("_", " ")
        n = len(mats)

    return pd.DataFrame(avg, index=models, columns=models), label, n


# ── Plotting ───────────────────────────────────────────────────────────────────

def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _cluster_colors(Z: np.ndarray, n: int, method: str) -> list[str]:
    """Assign a distinct color to each flat cluster."""
    # cut at ~50% of max linkage height to form natural groups
    labels = fcluster(Z, t=0.5 * Z[:, 2].max(), criterion="distance")
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
               "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    return [palette[(lbl - 1) % len(palette)] for lbl in labels]


def plot_network(ax: plt.Axes, corr_df: pd.DataFrame,
                 metric_label: str) -> None:
    """Draw a weighted network graph: edge width/color = correlation strength.
    All visual parameters scale automatically with the number of models."""
    models = corr_df.index.tolist()
    n = len(models)

    # ── Scale factors (anchored to n=5) ──────────────────────────────────────
    s = 5 / max(n, 5)                           # 1.0 at n=5, shrinks for larger n
    node_size       = max(300,  int(2400 * s))
    node_lw         = max(1.2,  3.0  * s)
    label_fontsize  = max(6.0,  13.5 * s)
    label_pad       = max(0.15, 0.4  * s)
    edge_lw_base    = max(0.5,  1.5  * s)
    edge_lw_range   = max(2.0,  10.0 * s)
    elabel_fontsize = max(4.5,  7.5  * s)
    show_edge_labels = n <= 10              # suppress at high density

    # label radius: gap between node and label grows a little for large n
    # so overlapping text from nearby nodes doesn't merge into the node
    label_r = 1.28 + 0.008 * max(0, n - 5)

    # ── Graph ─────────────────────────────────────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from(models)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(models[i], models[j], weight=corr_df.iloc[i, j])

    # n-gon layout — nodes equally spaced on the unit circle, top starts at 12 o'clock
    angles = [np.pi / 2 + 2 * np.pi * i / n for i in range(n)]
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(models, angles)}

    # Distinct saturated colors — palette repeats for > 10 models
    _palette = [
        "#E74C3C", "#3498DB", "#F1C40F", "#2ECC71", "#9B59B6",
        "#E67E22", "#1ABC9C", "#E91E8C", "#00BCD4", "#FF5722",
    ]
    node_colors = [_palette[i % len(_palette)] for i in range(n)]

    # ── Edges ─────────────────────────────────────────────────────────────────
    edges   = list(G.edges(data=True))
    weights = np.array([d["weight"] for _, _, d in edges])
    norm    = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap    = matplotlib.colormaps["RdYlGn"]

    w_range    = max(weights.max() - weights.min(), 1e-9)
    edge_widths = edge_lw_base + edge_lw_range * (weights - weights.min()) / w_range
    edge_colors = [cmap(norm(w)) for w in weights]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           edge_color=edge_colors, alpha=0.75)

    if show_edge_labels:
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=elabel_fontsize, font_color="#333333", label_pos=0.42,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                  "ec": "none", "alpha": 0.82},
        )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, edgecolors="#ffffff",
                           linewidths=node_lw, alpha=0.95)

    # ── Labels pushed radially outward ────────────────────────────────────────
    model_color = dict(zip(models, node_colors))
    for node, (x, y) in pos.items():
        mag = max(np.hypot(x, y), 1e-9)
        lx, ly = x * label_r / mag, y * label_r / mag
        ha = "left" if x > 0.05 else ("right" if x < -0.05 else "center")
        ax.text(
            lx, ly, node,
            ha=ha, va="center",
            fontsize=label_fontsize, fontweight="bold", color=model_color[node],
            bbox={"boxstyle": f"round,pad={label_pad}", "fc": "white",
                  "ec": model_color[node], "alpha": 0.95, "linewidth": 1.2},
            zorder=5,
        )

    # Axes bounds: fixed margin beyond label_r so text is never clipped
    ax.set_xlim(-label_r - 0.65, label_r + 0.65)
    ax.set_ylim(-label_r - 0.40, label_r + 0.40)
    ax.axis("off")


def plot_heatmap(ax: plt.Axes, corr_df: pd.DataFrame, leaf_order: list[int],
                 metric_label: str) -> None:
    ordered = [corr_df.index[i] for i in leaf_order]
    corr_ord = corr_df.loc[ordered, ordered]

    sns.heatmap(
        corr_ord, ax=ax,
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.65, "label": metric_label},
        square=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)


def plot_all(corr_df: pd.DataFrame, metric_label: str, dataset: str,
             method: str, n_sources: int, save_path: str | None) -> None:
    from scipy.cluster.hierarchy import leaves_list

    short = {m: _short(m) for m in corr_df.index}
    corr_df = corr_df.rename(index=short, columns=short)
    n = len(corr_df)

    dist = np.clip(1.0 - corr_df.values, 0, None)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist), method=method)
    node_colors = _cluster_colors(Z, n, method)
    leaf_order  = list(leaves_list(Z))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#f7f7f7",
        "figure.facecolor": "#ffffff",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig_h = max(9.6, 6 + n * 0.5)   # grows with number of models
    fig = plt.figure(figsize=(20.4, fig_h))
    fig.patch.set_facecolor("#ffffff")

    fig.text(0.5, 0.97,
             "LLM Judge Bench Results",
             ha="center", va="top", fontsize=16, fontweight="bold", color="#1a1a1a")

    gs = gridspec.GridSpec(1, 2, wspace=0.25, width_ratios=[1.3, 1],
                           left=0.02, right=0.97, top=0.92, bottom=0.08)

    ax_net  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    ax_net.set_facecolor("#f7f7f7")
    plot_network(ax_net, corr_df, metric_label)
    plot_heatmap(ax_heat, corr_df, leaf_order, metric_label)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    outputs_dir = ROOT / args.outputs

    if not outputs_dir.is_dir():
        sys.exit(f"Outputs directory not found: {outputs_dir}")

    corr_df, metric_label, n_sources = weighted_avg_corr(outputs_dir, args.dataset, args.metric)

    print(f"Metric: {metric_label}")
    print(f"Matrices averaged per metric: {n_sources}")
    print(f"Models: {corr_df.index.tolist()}")

    plot_all(
        corr_df,
        metric_label=metric_label,
        dataset=args.dataset,
        method=args.method,
        n_sources=n_sources,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
