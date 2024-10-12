import torch
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Tuple
import os


def plot_datasets_and_classifiers(
    datasets: List[Tuple[Tensor, Tensor]],
    classifiers: List[Tuple[Tensor, float]],
    save_path: str,
    plot_fraction: float = 1.0,
) -> None:
    """
    Plots multiple 2D datasets, classifiers, and transformations between datasets.

    Parameters:
    - datasets: List of tuples (X, y), where:
        - X: Tensor of shape (n_samples, 2), features of the dataset.
        - y: Tensor of shape (n_samples,) or (n_samples, 1), labels (-1 or 1).
        The datasets are ordered such that each point X_i in dataset m + 1 is the transformed
        version of X_i in dataset m.

    - classifiers: List of tuples (w, b), where:
        - w: Tensor of shape (2,), weight vector of the linear classifier.
        - b: float, bias term of the classifier.

    - save_path: str, the file path where the plot will be saved.

    - plot_fraction: float, between 0 and 1, the fraction of datapoints to plot.

    The function plots:
    - Datasets with datapoints colored according to labels and dataset index,
      with colors getting darker for subsequent datasets.
    - Linear classifiers as lines, colored in shades of green, darker for each classifier.
    - Arrows from each point in dataset m to the corresponding point in dataset m + 1
      if they are not the same.

    Returns:
    - None
    """
    k = len(classifiers)
    assert (
        len(datasets) == k + 1
    ), "Number of datasets must be k + 1 where k is the number of classifiers."
    assert 0 < plot_fraction <= 1, "plot_fraction must be between 0 and 1."
    # Set up colors for negative labels (reds), positive labels (blues), and classifiers (greens)
    reds_cmap = cm.get_cmap("Reds")
    blues_cmap = cm.get_cmap("Blues")
    greens_cmap = cm.get_cmap("Greens")

    reds = reds_cmap(torch.linspace(0.4, 0.9, k + 1).numpy())  # Light to dark reds
    blues = blues_cmap(torch.linspace(0.4, 0.9, k + 1).numpy())  # Light to dark blues
    greens = greens_cmap(
        torch.linspace(0.4, 0.9, k).numpy()
    )  # Light to dark greens for classifiers

    plt.figure(figsize=(10, 8))

    # Determine plotting limits
    all_X = torch.cat([X for X, _ in datasets], dim=0)
    x_min, x_max = all_X[:, 0].min().item() - 1, all_X[:, 0].max().item() + 1
    y_min, y_max = all_X[:, 1].min().item() - 1, all_X[:, 1].max().item() + 1

    # Determine indices to plot based on plot_fraction
    n_samples = datasets[0][0].shape[0]

    num_points_to_plot = max(1, int(n_samples * plot_fraction))  # At least one point
    indices = torch.arange(n_samples)
    if num_points_to_plot < n_samples:
        indices = indices[torch.randperm(n_samples)[:num_points_to_plot]]
        indices, _ = torch.sort(
            indices
        )  # Optional: sort indices for consistent ordering

    # Plot datasets
    for idx, (X, y) in enumerate(datasets):
        color_neg = reds[idx]
        color_pos = blues[idx]

        # Select subset of data to plot
        X_subset = X[indices]
        y_subset = y[indices]

        # Flatten labels if necessary
        y_flat = y_subset.view(-1)

        # Plot negative points
        X_neg = X_subset[y_flat == -1]
        plt.scatter(
            X_neg[:, 0].numpy(),
            X_neg[:, 1].numpy(),
            color=color_neg,
            label=f"Dataset {idx} Negative",
            alpha=0.6,
        )

        # Plot positive points
        X_pos = X_subset[y_flat == 1]
        plt.scatter(
            X_pos[:, 0].numpy(),
            X_pos[:, 1].numpy(),
            color=color_pos,
            label=f"Dataset {idx} Positive",
            alpha=0.6,
        )

    # Draw arrows between datasets
    for m in range(len(datasets) - 1):
        X_current, _ = datasets[m]
        X_next, _ = datasets[m + 1]

        # Ensure that the datasets have the same number of points
        assert (
            X_current.shape == X_next.shape
        ), f"Datasets at index {m} and {m + 1} do not have the same shape."

        # Select the same subset of data
        X_current_subset = X_current[indices]
        X_next_subset = X_next[indices]

        delta_movement = X_next_subset - X_current_subset
        deltas_nonzero = torch.any(delta_movement != 0, dim=1)

        for i in range(len(X_current_subset)):
            if deltas_nonzero[i]:
                plt.arrow(
                    X_current_subset[i, 0].item(),
                    X_current_subset[i, 1].item(),  # Starting point
                    delta_movement[i, 0].item(),
                    delta_movement[i, 1].item(),  # Movement
                    head_width=0.05,
                    head_length=0.1,
                    fc="gray",
                    ec="gray",
                    alpha=0.5,
                    length_includes_head=True,
                )

    # Plot classifiers
    x_vals = torch.linspace(x_min, x_max, 400)
    for idx, (w, b) in enumerate(classifiers):
        color = greens[idx]

        # Ensure w[1] is not zero to avoid division by zero
        if w[1] != 0:
            y_vals = (-w[0] * x_vals - b) / w[1]
            plt.plot(
                x_vals.numpy(),
                y_vals.numpy(),
                color=color,
                label=f"Classifier {idx}",
                linewidth=2,
            )
        else:
            # Vertical line at x = -b / w[0]
            x_line = -b / w[0]
            plt.axvline(
                x=x_line.item(), color=color, label=f"Classifier {idx}", linewidth=2
            )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Datasets, Classifiers, and Transformations Between Datasets")
    plt.grid(True)

    # Save the plot to the specified directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
