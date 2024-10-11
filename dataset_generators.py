import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def generate_circular_2d_dataset(
    num_samples: int, radius: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """
    Generates a 2D dataset with a circular decision boundary.

    Args:
        num_samples (int): Number of samples to generate.
        radius (float, optional): Radius of the circle defining the decision boundary. Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor]: Features tensor of shape (num_samples, 2) and labels tensor of shape (num_samples, 1).
    """
    # Generate random 2D points
    X = torch.randn(num_samples, 2)
    # Compute distance from origin
    radius_squared = X[:, 0] ** 2 + X[:, 1] ** 2
    # Labels: +1 if outside the circle, else -1
    y = torch.where(radius_squared > radius**2, torch.tensor(1.0), torch.tensor(-1.0))
    y = y.view(-1, 1)
    return X, y


def generate_spiral_2d_dataset(
    num_samples: int, noise: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """
    Generates a 2D spiral dataset.

    Args:
        num_samples (int): Total number of samples to generate.
        noise (float, optional): Standard deviation of Gaussian noise added to the data. Defaults to 0.5.

    Returns:
        Tuple[Tensor, Tensor]: Features tensor and labels tensor.
    """
    n = torch.sqrt(torch.rand(num_samples // 2)) * 780 * (2 * torch.pi) / 360
    d1x = -torch.cos(n) * n + torch.randn(num_samples // 2) * noise
    d1y = torch.sin(n) * n + torch.randn(num_samples // 2) * noise

    d2x = torch.cos(n) * n + torch.randn(num_samples // 2) * noise
    d2y = -torch.sin(n) * n + torch.randn(num_samples // 2) * noise

    X = torch.vstack([torch.hstack([d1x, d2x]), torch.hstack([d1y, d2y])]).T
    y = torch.hstack(
        [torch.ones(num_samples // 2), -torch.ones(num_samples // 2)]
    ).view(-1, 1)

    # Shuffle the dataset
    perm = torch.randperm(num_samples)
    X = X[perm]
    y = y[perm]

    return X, y


def generate_moons_dataset(
    num_samples: int, noise: float = 0.1
) -> Tuple[Tensor, Tensor]:
    """
    Generates a 2D two moons dataset.

    Args:
        num_samples (int): Total number of samples to generate.
        noise (float, optional): Standard deviation of Gaussian noise added to the data. Defaults to 0.1.

    Returns:
        Tuple[Tensor, Tensor]: Features tensor and labels tensor.
    """
    from sklearn.datasets import make_moons

    X_np, y_np = make_moons(n_samples=num_samples, noise=noise)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np * 2 - 1, dtype=torch.float32).view(
        -1, 1
    )  # Convert labels to -1 and 1
    return X, y


def generate_nd_dataset(num_samples: int, num_features: int) -> Tuple[Tensor, Tensor]:
    """
    Generates an N-dimensional dataset with a linear decision boundary.

    Args:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of features (dimensions).

    Returns:
        Tuple[Tensor, Tensor]: Features tensor of shape (num_samples, num_features) and labels tensor of shape (num_samples, 1).
    """
    X = torch.randn(num_samples, num_features)
    # Labels: +1 if sum of features > 0, else -1
    y = torch.sign(X.sum(dim=1))
    y[y == 0] = 1
    y = y.view(-1, 1)
    return X, y


def generate_gaussian_clusters_dataset(
    num_samples: int,
    mean_pos: float = 2.0,
    mean_neg: float = -2.0,
    std_dev: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Generates a dataset with two Gaussian clusters: one for positive labels and one for negative labels.

    Args:
        num_samples (int): Total number of samples to generate.
        mean_pos (float, optional): Mean of the positive class Gaussian distribution. Defaults to 2.0.
        mean_neg (float, optional): Mean of the negative class Gaussian distribution. Defaults to -2.0.
        std_dev (float, optional): Standard deviation of the Gaussian distributions. Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor]: Features tensor and labels tensor.
    """
    num_samples_per_class = num_samples // 2

    # Positive class samples
    X_pos = torch.randn(num_samples_per_class, 2) * std_dev + mean_pos
    y_pos = torch.ones(num_samples_per_class, 1)

    # Negative class samples
    X_neg = torch.randn(num_samples_per_class, 2) * std_dev + mean_neg
    y_neg = -torch.ones(num_samples_per_class, 1)

    # Combine datasets
    X = torch.vstack((X_pos, X_neg))
    y = torch.vstack((y_pos, y_neg))

    # Shuffle the dataset
    perm = torch.randperm(num_samples)
    X = X[perm]
    y = y[perm]

    return X, y


def add_feature_noise(X: Tensor, noise_level: float = 0.1) -> Tensor:
    noise = torch.randn_like(X) * noise_level
    return X + noise
