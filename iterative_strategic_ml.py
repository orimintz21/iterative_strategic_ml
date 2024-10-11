import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import strategic_ml as sml
from typing import Tuple, List

import dataset_generators as dg
import visualization as vis

class BCEWithLogitsLossPMOne(nn.Module):
    """Self-defined BCEWithLogitsLoss with target values in {-1, 1}."""
    def __init__(self):
        super(BCEWithLogitsLossPMOne, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.loss_fn(input, (target + 1) / 2)

class MSELossPOne(nn.Module):
    """Self-defined MSE loss with target values in {-1, 1}."""
    def __init__(self):
        super(MSELossPOne, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, input, target):
        return self.loss_fn(input, (target + 1) / 2)



def generate_initial_dataset(
    args: argparse.Namespace,
) -> Tuple[Tensor, Tensor]:
    num_samples = args.num_samples
    if args.dataset == "linear":
        if args.num_features is not None:
            X, y = dg.generate_nd_dataset(num_samples, num_features=args.num_features)
        else:
            X, y = dg.generate_nd_dataset(num_samples, 2)
    elif args.dataset == "gaussian":
        mean_pos = args.mean_pos if args.mean_pos is not None else 2.0
        mean_neg = args.mean_neg if args.mean_neg is not None else -2.0
        std_dev = args.std_dev if args.std_dev is not None else 1.0
        X, y = dg.generate_gaussian_clusters_dataset(
            num_samples, mean_pos, mean_neg, std_dev
        )
    elif args.dataset == "circular":
        if args.radius is not None:
            X, y = dg.generate_circular_2d_dataset(num_samples, radius=args.radius)
        else:
            X, y = dg.generate_circular_2d_dataset(num_samples)
    elif args.dataset == "spiral":
        if args.noise is not None:
            X, y = dg.generate_spiral_2d_dataset(num_samples, noise=args.noise)
        else:
            X, y = dg.generate_spiral_2d_dataset(num_samples)
    elif args.dataset == "moons":
        if args.noise is not None:
            X, y = dg.generate_moons_dataset(num_samples, noise=args.noise)
        else:
            X, y = dg.generate_moons_dataset(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.dataset != "linear":
        assert args.num_features == 2, "Only the linear dataset is not necessarily 2D."
    
    return X, y


def experiment(
    args: argparse.Namespace,
):

    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    X, y = generate_initial_dataset(args)
    datasets: List[Tuple[Tensor, Tensor]] = [(X, y)]
    num_features = X.shape[1]
    model = sml.LinearModel(num_features)
    loss_fn_str = args.loss_fn
    for iteration in range(args.num_iterations):
        model.fit(X, y)
        delta = model.compute_strategic_delta(X, y)
        X_prime = X + delta
        datasets.append((X_prime, y))
        X = X_prime



def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
