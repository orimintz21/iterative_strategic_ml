import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# Assuming strategic_ml is the library we've discussed
# Import the LinearStrategicDelta
from strategic_ml.gsc import LinearStrategicDelta


def plot_datasets_and_classifiers(
    datasets: List[Tuple[Tensor, Tensor]],
    deltas: List[LinearStrategicDelta],
    classifiers: List[Tuple[Tensor, float]],
    save_dir: str,
) -> None:
    """
    Plots multiple 2D datasets, classifiers, and delta-induced transformations.

    Parameters:
    - datasets: List of tuples (X, y), where:
        - X: Tensor of shape (n_samples, 2), features of the dataset.
        - y: Tensor of shape (n_samples,) or (n_samples, 1), labels (-1 or 1).
    - deltas: List of LinearStrategicDelta instances corresponding to datasets.
    - classifiers: List of tuples (w, b), where:
        - w: Tensor of shape (2,), weight vector of the linear classifier.
        - b: float, bias term of the classifier.
    - save_dir: Directory to save the plot.

    The function plots:
    - Datasets with datapoints colored according to labels and dataset index,
      with colors getting darker for subsequent datasets.
    - Linear classifiers as lines, colored in shades of green, darker for each classifier.
    - Arrows from original datapoints to their delta-transformed points (if they move).

    Returns:
    - None
    """
    k = len(classifiers)
    assert (
        len(datasets) == k 
    ), "Number of datasets must be k where k is the number of classifiers."
    assert (
        len(deltas) == k 
    ), "Number of deltas must be k  where k is the number of classifiers."

    reds_cmap = cm.get_cmap("Reds")
    blues_cmap = cm.get_cmap("Blues")
    greens_cmap = cm.get_cmap("Greens")

    reds = reds_cmap(torch.linspace(0.4, 0.9, k + 1).numpy())  # Light to dark reds, add one for the last transformation
    blues = blues_cmap(torch.linspace(0.4, 0.9, k + 1).numpy())  # Light to dark blues, add one for the last transformation
    greens = greens_cmap(
        torch.linspace(0.4, 0.9, k).numpy()
    )  # Light to dark greens for classifiers

    plt.figure(figsize=(10, 8))

    # Determine plotting limits
    all_X = torch.cat([X for X, _ in datasets], dim=0)
    x_min, x_max = all_X[:, 0].min().item() - 1, all_X[:, 0].max().item() + 1
    y_min, y_max = all_X[:, 1].min().item() - 1, all_X[:, 1].max().item() + 1

    # Plot datasets and deltas
    for idx, ((X, y), delta) in enumerate(zip(datasets, deltas)):
        color_neg = reds[idx]
        color_pos = blues[idx]

        # Flatten labels if necessary
        y_flat = y.view(-1)

        # Plot negative points
        X_neg = X[y_flat == -1]
        plt.scatter(
            X_neg[:, 0].numpy(),
            X_neg[:, 1].numpy(),
            color=color_neg,
            label=f"Dataset {idx} Negative",
            alpha=0.6,
        )

        # Plot positive points
        X_pos = X[y_flat == 1]
        plt.scatter(
            X_pos[:, 0].numpy(),
            X_pos[:, 1].numpy(),
            color=color_pos,
            label=f"Dataset {idx} Positive",
            alpha=0.6,
        )

        # Apply delta to X
        # Since we're at the end of the test, we can call delta directly
        # Assume delta operates as a callable: delta(X, y)
        X_delta = delta(X, y)

        # For points where delta changes the point, draw arrows
        delta_movement = X_delta - X
        deltas_nonzero = torch.any(delta_movement != 0, dim=1)
        for i in range(len(X)):
            if deltas_nonzero[i]:
                plt.arrow(
                    X[i, 0].item(),
                    X[i, 1].item(),  # Starting point (original data point)
                    delta_movement[i, 0].item(),
                    delta_movement[i, 1].item(),  # Delta movement
                    head_width=0.05,
                    head_length=0.1,
                    fc="gray",
                    ec="gray",
                    alpha=0.5,
                    length_includes_head=True,
                )

        # If this is the last delta, plot the transformed points
        if idx == k - 1:
            color_neg = reds[idx + 1]
            color_pos = blues[idx + 1]

            # Flatten labels if necessary
            y_flat = y.view(-1)

            # Plot negative points
            X_neg = X_delta[y_flat == -1]
            plt.scatter(
                X_neg[:, 0].numpy(),
                X_neg[:, 1].numpy(),
                color=color_neg,
                label=f"Dataset {idx} Negative",
                alpha=0.6,
            )

            # Plot positive points
            X_pos = X_delta[y_flat == 1]
            plt.scatter(
                X_pos[:, 0].numpy(),
                X_pos[:, 1].numpy(),
                color=color_pos,
                label=f"Dataset {idx} Positive",
                alpha=0.6,
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
    plt.title("Datasets, Classifiers, and Delta Transformations")
    plt.grid(True)
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "datasets_classifiers_deltas.png"))
