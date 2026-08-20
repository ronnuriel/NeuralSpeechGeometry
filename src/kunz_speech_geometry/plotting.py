"""Small plotting helpers used by the notebooks."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLORS = {
    "attempted_vocalized": "#D1495B",
    "passive_listening": "#00798C",
}


def plot_scree(explained_variance_ratio: np.ndarray):
    """Plot individual and cumulative explained variance."""
    ratios = np.asarray(explained_variance_ratio)
    pcs = np.arange(1, ratios.size + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pcs, 100 * ratios, color="#5B6C8F", alpha=0.8, label="individual")
    ax.plot(pcs, 100 * np.cumsum(ratios), "o-", color="#D1495B", label="cumulative")
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)", xticks=pcs)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_condition_trajectories(
    scores: np.ndarray,
    labels: np.ndarray,
    time_ms: np.ndarray,
    *,
    colors: Mapping[str, str] = DEFAULT_COLORS,
):
    """Plot condition-mean PC1/PC2 trajectories with onset and endpoint markers."""
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for condition in sorted(np.unique(labels)):
        mean_trajectory = scores[labels == condition].mean(axis=0)
        color = colors.get(condition)
        ax.plot(
            mean_trajectory[:, 0],
            mean_trajectory[:, 1],
            color=color,
            linewidth=2.2,
            label=condition.replace("_", " "),
        )
        onset_index = int(np.argmin(np.abs(time_ms)))
        ax.scatter(
            mean_trajectory[onset_index, 0],
            mean_trajectory[onset_index, 1],
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            "0 ms",
            mean_trajectory[onset_index, :2],
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axhline(0, color="0.88", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.88", linewidth=0.8, zorder=0)
    ax.set(xlabel="Shared PC1", ylabel="Shared PC2", title="Condition-mean neural trajectories")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_time_resolved_distance(
    time_ms: np.ndarray,
    distance: np.ndarray,
    analysis_window_ms: tuple[float, float] | list[float],
):
    """Plot the shared-space centroid distance across time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ms, distance, color="#5B6C8F", linewidth=2)
    ax.axvline(0, color="0.25", linestyle="--", linewidth=1)
    ax.axvspan(*analysis_window_ms, color="#E9C46A", alpha=0.2, label="analysis window")
    ax.set(
        xlabel="Time from alignment event (ms)",
        ylabel="Centroid distance (PC units)",
        title="Attempted vs passive distance over time",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_word_centroids(word_pca):
    """Plot word centroids, connecting the same word across two conditions."""
    metadata = word_pca.metadata
    scores = word_pca.scores
    conditions = sorted(metadata["condition"].unique())
    words = sorted(metadata["word"].unique())
    markers = {conditions[0]: "o", conditions[1]: "s"}
    color_values = plt.get_cmap("tab10")(np.linspace(0, 1, len(words)))
    word_colors = dict(zip(words, color_values, strict=True))

    fig, ax = plt.subplots(figsize=(8, 6))
    for word in words:
        indices = metadata.index[metadata["word"] == word].to_numpy()
        if indices.size == 2:
            ax.plot(scores[indices, 0], scores[indices, 1], color=word_colors[word], alpha=0.45)
        for index in indices:
            condition = metadata.loc[index, "condition"]
            ax.scatter(
                scores[index, 0],
                scores[index, 1],
                color=word_colors[word],
                marker=markers[condition],
                s=85,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.annotate(word, scores[index, :2], xytext=(5, 4), textcoords="offset points")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[condition],
            color="none",
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=9,
            label=condition.replace("_", " "),
        )
        for condition in conditions
    ]
    ax.set(
        xlabel=f"Word-centroid PC1 ({100 * word_pca.explained_variance_ratio[0]:.1f}%)",
        ylabel=f"Word-centroid PC2 ({100 * word_pca.explained_variance_ratio[1]:.1f}%)",
        title="Shared condition × word geometry",
    )
    ax.legend(handles=handles, frameon=False, title="Condition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_word_rdms(conditions, words, rdms):
    """Plot two word RDMs and their signed difference on comparable scales."""
    max_distance = float(np.nanmax(rdms))
    difference = rdms[0] - rdms[1]
    diff_limit = float(np.max(np.abs(difference))) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for index, condition in enumerate(conditions):
        image = axes[index].imshow(rdms[index], vmin=0, vmax=max_distance, cmap="viridis")
        axes[index].set_title(condition.replace("_", " "))
        axes[index].set_xticks(range(len(words)), words, rotation=45, ha="right")
        axes[index].set_yticks(range(len(words)), words)
    fig.colorbar(image, ax=axes[:2], shrink=0.82, label="Euclidean distance")

    diff_image = axes[2].imshow(
        difference,
        vmin=-diff_limit,
        vmax=diff_limit,
        cmap="coolwarm",
    )
    axes[2].set_title(f"{conditions[0]} minus\n{conditions[1]}")
    axes[2].set_xticks(range(len(words)), words, rotation=45, ha="right")
    axes[2].set_yticks(range(len(words)), words)
    fig.colorbar(diff_image, ax=axes[2], shrink=0.82, label="Distance difference")
    return fig, axes


def plot_permutation(null_distribution: np.ndarray, observed: float, p_value: float):
    """Plot a permutation null distribution and observed statistic."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_distribution, bins=30, color="#9BA7C0", edgecolor="white")
    ax.axvline(observed, color="#D1495B", linewidth=2.2, label=f"observed; p={p_value:.4f}")
    ax.set(xlabel="Permuted centroid distance", ylabel="Count", title="Stratified trial-label null")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax
