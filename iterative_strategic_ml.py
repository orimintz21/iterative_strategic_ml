import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import strategic_ml as sml
from typing import Tuple, List

import dataset_generators as dg
import visualization as vis


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_features", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="gaussian")
    parser.add_argument("--mean_pos", type=float, default=2.0)
    parser.add_argument("--mean_neg", type=float, default=-2.0)
    parser.add_argument("--std_dev", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    # training
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    # strategic learning
    parser.add_argument("--start_cost_weight", type=float, default=1)
    parser.add_argument("--cost_weight_multiplier", type=float, default=1.5)
    parser.add_argument("--cost_weight_addend", type=float, default=0)
    parser.add_argument("--loss_fn", type=str, default="bce")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--linear_regulation_fn", type=str, default="l1")
    parser.add_argument("--linear_regulation_strength", type=float, default=0.01)
    parser.add_argument("--elastic_ratio", type=float, default=0.5)
    # other
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--plot_fraction", type=float, default=0.1)
    parser.add_argument("--visualize", action="store_true", default=True)

    args = parser.parse_args()
    print("args", args)
    return args


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

    # Shuffle the dataset
    perm = torch.randperm(num_samples)
    X = X[perm]
    y = y[perm]
    return X, y


def split_train_val_test(
    X: Tensor, y: Tensor, val_ratio: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_samples = X.shape[0]
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val
    perm = torch.randperm(num_samples)
    X = X[perm]
    y = y[perm]
    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = X[num_train:], y[num_train:]
    return X_train, y_train, X_val, y_val


def experiment(
    args: argparse.Namespace,
):
    if args.visualize:
        assert (
            args.num_features == 2
        ), "Visualization is only supported for 2D datasets."
    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    X, y = generate_initial_dataset(args)

    X_train, y_train, X_val, y_val = split_train_val_test(X, y, args.val_ratio)

    datasets: List[Tuple[Tensor, Tensor]] = [(X_train, y_train)]
    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    validation_dataloader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    # We don't use the test
    test_dataloader = train_dataloader

    classifiers: List[Tuple[Tensor, float]] = []
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    cost_weight: float = args.start_cost_weight
    num_features = X.shape[1]
    model = sml.LinearModel(num_features)
    cost = sml.CostNormL2(dim=1)
    delta = sml.LinearStrategicDelta(
        strategic_model=model, cost=cost, cost_weight=cost_weight
    )
    if args.loss_fn == "bce":
        loss_fn = BCEWithLogitsLossPMOne()
    elif args.loss_fn == "mse":
        loss_fn = MSELossPOne()
    elif args.loss_fn == "hinge":
        loss_fn = nn.HingeEmbeddingLoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    if args.optimizer == "sgd":
        optimizer = optim.SGD
    elif args.optimizer == "adam":
        optimizer = optim.Adam
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    reg_fn = None

    if args.linear_regulation_fn is not None:
        assert (
            args.linear_regulation_strength is not None
        ), "L1 regularization strength must be specified."
        if args.linear_regulation_fn == "l1":
            reg_fn = sml.LinearL1Regularization(args.linear_regulation_strength)
        elif args.linear_regulation_fn == "l2":
            reg_fn = sml.LinearL2Regularization(args.linear_regulation_strength)
        elif args.linear_regulation_fn == "elastic":
            assert (
                args.elastic_ratio is not None
            ), "Elastic net ratio must be specified."
            reg_fn = sml.LinearElasticNetRegularization(
                args.linear_regulation_strength, args.elastic_ratio
            )
        else:
            raise ValueError(
                f"Unknown regularization function: {args.linear_regulation_fn}"
            )

    if reg_fn is not None:
        model_suit = sml.ModelSuit(
            model=model,
            delta=delta,
            loss_fn=loss_fn,
            training_params={
                "optimizer": optimizer,
                "lr": args.lr,
            },
            train_loader=train_dataloader,
            validation_loader=validation_dataloader,
            test_loader=test_dataloader,
            linear_regularization=[reg_fn],
        )
    else:
        model_suit = sml.ModelSuit(
            model=model,
            delta=delta,
            loss_fn=loss_fn,
            training_params={
                "optimizer": optimizer,
                "lr": args.lr,
            },
            train_loader=train_dataloader,
            validation_loader=validation_dataloader,
            test_loader=test_dataloader,
        )

    logging_dir = os.path.join(save_dir, "logs")

    for iteration in range(args.num_iterations):
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=CSVLogger(
                logging_dir,
                name=f"iter{iteration}",
            ),
        )
        trainer.fit(model_suit)
        # Save the classifier

        if args.visualize:
            w, b = model.get_weight_and_bias()
            w = w.view(-1)
            b = b.item()
            classifiers.append((w, b))

        # Create the next dataset
        X_train = model_suit.delta(X_train, y_train)
        X_val = model_suit.delta(X_val, y_val)

        if args.visualize:
            datasets.append((X_train, y_train))

        train_dataloader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        validation_dataloader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Update the model suit
        model_suit.train_loader = train_dataloader
        model_suit.validation_loader = validation_dataloader

        # Update the cost weight
        cost_weight *= args.cost_weight_multiplier
        cost_weight += args.cost_weight_addend
        model_suit.delta.cost_weight = cost_weight

    # Visualize the classifiers
    if args.visualize:
        vis.plot_datasets_and_classifiers(
            datasets=datasets,
            classifiers=classifiers,
            save_path=vis_dir,
            plot_fraction=args.plot_fraction,
        )


def main():
    args = parse_args()
    experiment(args)


if __name__ == "__main__":
    main()
