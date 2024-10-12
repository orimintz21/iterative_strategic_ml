import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from typing import List, Tuple
import os
import textwrap


def plot_datasets_and_classifiers(
    datasets: List[Tuple[Tensor, Tensor]],
    classifiers: List[Tuple[Tensor, float]],
    save_path: str,
    dataset_names: str,
    cost_start: float,
    cost_multiplier: float,
    cost_add: float,
    plot_fraction: float = 1.0,
    title_width: int = 80,  # Max characters per line in the title
) -> None:
    k = len(classifiers)
    assert (
        len(datasets) == k + 1
    ), "Number of datasets must be k + 1 where k is the number of classifiers."
    assert 0 < plot_fraction <= 1, "plot_fraction must be between 0 and 1."

    reds_cmap = cm.get_cmap("Reds")
    blues_cmap = cm.get_cmap("Blues")
    greens_cmap = cm.get_cmap("Greens")

    reds = reds_cmap(torch.linspace(0.4, 0.9, k + 1).numpy())
    blues = blues_cmap(torch.linspace(0.4, 0.9, k + 1).numpy())
    greens = greens_cmap(torch.linspace(0.4, 0.9, k).numpy())

    fig, ax = plt.subplots(figsize=(10, 8))

    all_X = torch.cat([X for X, _ in datasets], dim=0)
    x_min, x_max = all_X[:, 0].min().item() - 1, all_X[:, 0].max().item() + 1
    y_min, y_max = all_X[:, 1].min().item() - 1, all_X[:, 1].max().item() + 1

    n_samples = datasets[0][0].shape[0]
    num_points_to_plot = max(1, int(n_samples * plot_fraction))
    indices = torch.arange(n_samples)
    if num_points_to_plot < n_samples:
        indices = indices[torch.randperm(n_samples)[:num_points_to_plot]]
        indices, _ = torch.sort(indices)

    for idx, (X, y) in enumerate(datasets):
        color_neg = reds[idx]
        color_pos = blues[idx]

        X_subset = X[indices]
        y_subset = y[indices].view(-1)

        X_neg = X_subset[y_subset == -1]
        ax.scatter(X_neg[:, 0].numpy(), X_neg[:, 1].numpy(), color=color_neg, alpha=0.6)

        X_pos = X_subset[y_subset == 1]
        ax.scatter(X_pos[:, 0].numpy(), X_pos[:, 1].numpy(), color=color_pos, alpha=0.6)

    for m in range(len(datasets) - 1):
        X_current, _ = datasets[m]
        X_next, _ = datasets[m + 1]
        assert (
            X_current.shape == X_next.shape
        ), f"Datasets at index {m} and {m + 1} do not have the same shape."

        X_current_subset = X_current[indices]
        X_next_subset = X_next[indices]
        delta_movement = X_next_subset - X_current_subset
        deltas_nonzero = torch.any(delta_movement != 0, dim=1)

        for i in range(len(X_current_subset)):
            if deltas_nonzero[i]:
                ax.arrow(
                    X_current_subset[i, 0].item(),
                    X_current_subset[i, 1].item(),
                    delta_movement[i, 0].item(),
                    delta_movement[i, 1].item(),
                    head_width=0.05,
                    head_length=0.1,
                    fc="gray",
                    ec="gray",
                    alpha=0.5,
                    length_includes_head=True,
                )

    x_vals = torch.linspace(x_min, x_max, 400)
    for idx, (w, b) in enumerate(classifiers):
        color = greens[idx]
        if w[1] != 0:
            y_vals = (-w[0] * x_vals - b) / w[1]
            ax.plot(x_vals.numpy(), y_vals.numpy(), color=color, linewidth=2)
        else:
            x_line = -b / w[0]
            ax.axvline(x=x_line.item(), color=color, linewidth=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Wrapping the title
    full_title = (
        f"{k} Iterative Strategic Classification on {dataset_names} Dataset\n"
        f"Cost: {cost_multiplier}^i * {cost_start} + i * {cost_add}"
    )
    wrapped_title = "\n".join(textwrap.wrap(full_title, width=title_width))
    ax.set_title(wrapped_title, fontsize=20, loc="center")  # Larger title

    ax.set_xlabel("Feature 1", fontsize=16)  # Larger x-axis label
    ax.set_ylabel("Feature 2", fontsize=16)  # Larger y-axis label
    ax.grid(True)

    # Custom legend with color patches
    legend_elements = [
        Patch(facecolor=greens[-1], edgecolor="black", label="Classifiers"),
        Patch(facecolor=blues[-1], edgecolor="black", label="Positive Labels"),
        Patch(facecolor=reds[-1], edgecolor="black", label="Negative Labels"),
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
