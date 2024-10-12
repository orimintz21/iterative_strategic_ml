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


class NonLinearModel(nn.Module):
    def __init__(self, hidden_size: int):
        super(NonLinearModel, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples."
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=2,
        help="Number of features. Only used for the linear dataset, if you select other dataset don't change it.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaussian",
        help="Dataset to use: linear, gaussian, circular, spiral, moons.",
    )
    parser.add_argument(
        "--mean_pos",
        type=float,
        default=2.0,
        help="Mean of the positive cluster for the gaussian dataset.",
    )
    parser.add_argument(
        "--mean_neg",
        type=float,
        default=-2.0,
        help="Mean of the negative cluster for the gaussian dataset.",
    )
    parser.add_argument(
        "--std_dev",
        type=float,
        default=1.0,
        help="Standard deviation of the clusters for the gaussian dataset.",
    )
    parser.add_argument(
        "--data_radius_multiplier",
        type=float,
        default=1.0,
        help="Radius multiplier for the circular dataset.",
    )
    parser.add_argument(
        "--label_radius",
        type=float,
        default=None,
        help="Radius that indicates where the label is positive.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Noise level for the spiral and moons datasets.",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation set ratio."
    )
    # training
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloaders.",
    )
    # strategic learning
    parser.add_argument(
        "--start_cost_weight", type=float, default=1, help="Initial cost weight."
    )
    parser.add_argument(
        "--cost_weight_multiplier",
        type=float,
        default=1.5,
        help="Cost weight multiplier.",
    )
    parser.add_argument(
        "--cost_weight_addend", type=float, default=0, help="Cost weight addend."
    )
    parser.add_argument(
        "--loss_fn", type=str, default="bce", help="Loss function: bce, mse, hinge."
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer: sgd, adam, adagrad."
    )
    parser.add_argument(
        "--linear_regulation_fn",
        type=str,
        default="l1",
        help="Linear regulation function: l1, l2, elastic.",
    )
    parser.add_argument(
        "--linear_regulation_strength",
        type=float,
        default=0.01,
        help="Linear regulation strength.",
    )
    parser.add_argument(
        "--elastic_ratio", type=float, default=0.5, help="Elastic net ratio."
    )
    # non-linear model
    parser.add_argument(
        "--non_linear", action="store_true", default=False, help="Use non-linear model."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=2,
        help="Hidden size for the non-linear model.",
    )
    parser.add_argument(
        "--non_linear_delta_optimizer",
        type=str,
        default="adam",
        help="Optimizer for the non-linear model.",
    )
    parser.add_argument(
        "--non_linear_delta_lr",
        type=float,
        default=0.001,
        help="Learning rate for the non-linear model.",
    )
    parser.add_argument(
        "--non_linear_delta_max_epochs",
        type=int,
        default=65,
        help="Maximum number of epochs for the non-linear model.",
    )
    parser.add_argument(
        "--non_linear_delta_temp",
        type=float,
        default=27.0,
        help="Temperature for the non-linear model.",
    )

    # other
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="Number of iterations."
    )
    parser.add_argument(
        "--save_dir", type=str, default="results", help="Directory to save the results."
    )
    parser.add_argument("--plot_name", type=str, default=None, help="Name of the plot.")
    parser.add_argument(
        "--plot_fraction",
        type=float,
        default=0.5,
        help="Fraction of datapoints to plot.",
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True, help="Visualize the results."
    )

    args = parser.parse_args()
    print("args", args)
    return args


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
        X, y = dg.generate_circular_2d_dataset(
            num_samples,
            data_radius_mul=args.data_radius_multiplier,
            label_radius=args.label_radius,
        )
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

    datasets: List[Tuple[Tensor, Tensor]] = [(X_val, y_val)]
    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
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
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    cost_weight: float = args.start_cost_weight
    num_features = X.shape[1]
    cost = sml.CostNormL2(dim=1)
    if args.non_linear:
        model = NonLinearModel(args.hidden_size)
        if args.non_linear_delta_optimizer == "adam":
            delta_optimizer_class = optim.Adam
        elif args.non_linear_delta_optimizer == "sgd":
            delta_optimizer_class = optim.SGD
        elif args.non_linear_delta_optimizer == "adagrad":
            delta_optimizer_class = optim.Adagrad
        else:
            raise ValueError(
                f"Unknown optimizer for non-linear model: {args.non_linear_delta_optimizer}"
            )
        dict_training_params = {
            "optimizer_class": delta_optimizer_class,
            "optimizer_params": {"lr": args.non_linear_delta_lr},
            "max_epochs": args.non_linear_delta_max_epochs,
            "temp": args.non_linear_delta_temp,
        }
        delta = sml.NonLinearStrategicDelta(
            strategic_model=model,
            cost=cost,
            cost_weight=cost_weight,
            training_params=dict_training_params,
        )

    else:
        model = sml.LinearModel(num_features)
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

    if args.linear_regulation_fn is not None and not args.non_linear:
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

        if args.visualize and not args.non_linear:
            w, b = model.get_weight_and_bias()
            w = w.view(-1)
            b = b.item()
            classifiers.append((w, b))

        # Create the next dataset
        X_train = model_suit.delta(X_train, y_train)
        X_val = model_suit.delta(X_val, y_val)

        if args.visualize:
            datasets.append((X_val, y_val))

        train_dataloader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
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
        if args.plot_name is not None:
            name = f"{args.plot_name}.png"
        else:
            name = f"num_iter_{args.num_iterations}_cost_{args.start_cost_weight}_multiplier_{args.cost_weight_multiplier}_add_{args.cost_weight_addend}_dataset_{args.dataset}"
            if args.non_linear:
                name += f"non_linear_hidden_{args.hidden_size}_temp_{args.non_linear_delta_temp}"
            name += ".png"

        vis_path = os.path.join(vis_dir, name)
        vis.plot_datasets_and_classifiers(
            datasets=datasets,
            classifiers=classifiers,
            save_path=vis_path,
            plot_fraction=args.plot_fraction,
            cost_start=args.start_cost_weight,
            cost_multiplier=args.cost_weight_multiplier,
            cost_add=args.cost_weight_addend,
            dataset_names=args.dataset,
        )


def main():
    args = parse_args()
    experiment(args)


if __name__ == "__main__":
    main()
