# External Imports
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from typing import Tuple, List, Optional
import strategic_ml as sml

# Internal Imports
import dataset_generators as dg
import visualization as vis


# ---------------------Class Declarations---------------------------------------#
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


# ------------------------Function Declarations---------------------------------#
def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples."
    )
    parser.add_argument(
        "--added_noise",
        type=float,
        default=0.0,
        help="Noise level to add to the dataset.",
    )
    parser.add_argument(
        "--test_samples", type=int, default=1000, help="Number of test samples."
    )
    parser.add_argument(
        "--test_added_noise",
        type=float,
        default=0.0,
        help="Noise level to add to the test dataset.",
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
        default=1.0,
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
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="Number of epochs to pretrain the model.",
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

    # In the dark
    parser.add_argument(
        "--itd", action="store_true", default=False, help="Use 'in the dark'."
    )
    parser.add_argument(
        "--itd_start_cost_weight",
        type=float,
        default=1,
        help="Initial cost weight for 'in the dark' users.",
    )
    parser.add_argument(
        "--itd_cost_weight_multiplier",
        type=float,
        default=1.5,
        help="Cost weight multiplier for 'in the dark' users.",
    )
    parser.add_argument(
        "--itd_cost_weight_addend",
        type=float,
        default=0,
        help="Cost weight addend for 'in the dark' users.",
    )
    parser.add_argument(
        "--itd_cost_override",
        action="store_true",
        default=True,
        help="Override the cost weight for 'in the dark' users with the train cost weight.",
    )
    parser.add_argument(
        "--train_val_update_itd",
        action="store_true",
        default=False,
        help="Update the training and validation using the itd delta and not the training delta.",
    )
    parser.add_argument(
        "--model_learn_test_percentage",
        type=float,
        default=0,
        help="Percentage of the test set to use for training the model after each iteration.",
    )
    parser.add_argument(
        "--test_train_max_epochs",
        type=int,
        default=20,
        help="Maximum number of epochs for training the model on the test set (used only if the model learn test percentage is larger than 0).",
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
    parser.add_argument(
        "--add_uniq_id",
        action="store_true",
        default=False,
        help="Add unique id to the save directory.",
    )

    args = parser.parse_args()
    print("args", args)
    return args


def generate_initial_dataset(
    args: argparse.Namespace,
    test: bool = False,
) -> Tuple[Tensor, Tensor]:
    num_samples = args.test_samples if test else args.num_samples

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

    if test:
        X = dg.add_feature_noise(X, args.test_added_noise)
    else:
        X = dg.add_feature_noise(X, args.added_noise)
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


def get_regularization_fn(args: argparse.Namespace):
    reg_fn = None
    if args.linear_regulation_fn is None or args.non_linear:
        return None

    assert (
        args.linear_regulation_strength is not None
    ), "L1 regularization strength must be specified."
    if args.linear_regulation_fn == "l1":
        reg_fn = sml.LinearL1Regularization(args.linear_regulation_strength)
    elif args.linear_regulation_fn == "l2":
        reg_fn = sml.LinearL2Regularization(args.linear_regulation_strength)
    elif args.linear_regulation_fn == "elastic":
        assert args.elastic_ratio is not None, "Elastic net ratio must be specified."
        reg_fn = sml.LinearElasticNetRegularization(
            args.linear_regulation_strength, args.elastic_ratio
        )
    else:
        raise ValueError(
            f"Unknown regularization function: {args.linear_regulation_fn}"
        )
    return reg_fn


def visualize_datasets_and_classifiers(
    args: argparse.Namespace,
    datasets: List[Tuple[Tensor, Tensor]],
    classifiers: List[Tuple[Tensor, float]],
    itd_classifiers: List[Tuple[Tensor, float]],
    experiment_name: str,
    save_dir: str,
):
    if not args.visualize:
        return
    name = args.plot_name
    if args.plot_name is None:
        name = f"visualization_{experiment_name}"

    name += ".png"

    vis_path = os.path.join(save_dir, name)
    vis.plot_datasets_and_classifiers(
        datasets=datasets,
        classifiers=classifiers,
        itd_classifiers=itd_classifiers,
        save_path=vis_path,
        plot_fraction=args.plot_fraction,
        cost_start=args.start_cost_weight,
        cost_multiplier=args.cost_weight_multiplier,
        cost_add=args.cost_weight_addend,
        dataset_names=args.dataset,
    )


def get_optimizer(optimizer: str):
    if optimizer == "sgd":
        return optim.SGD
    elif optimizer == "adam":
        return optim.Adam
    elif optimizer == "adagrad":
        return optim.Adagrad
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def get_loss_fn(loss_fn: str):
    if loss_fn == "bce":
        return BCEWithLogitsLossPMOne()
    elif loss_fn == "mse":
        return MSELossPOne()
    elif loss_fn == "hinge":
        return nn.HingeEmbeddingLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def output_experiment(
    args: argparse.Namespace,
    datasets: List[Tuple[Tensor, Tensor]],
    classifiers: List[Tuple[Tensor, float]],
    itd_classifiers: List[Tuple[Tensor, float]],
    results: List[Tuple[float, float, float]],
):
    experiment_name = f"dataset_{args.dataset}_linear_{not args.non_linear}_cost_{args.start_cost_weight}_multiplier_{args.cost_weight_multiplier}_add_{args.cost_weight_addend}"
    if args.itd:
        experiment_name += f"_itd_cost_{args.itd_start_cost_weight}_multiplier_{args.itd_cost_weight_multiplier}_add_{args.itd_cost_weight_addend}"
    if args.add_uniq_id:
        uniq_id = torch.randint(0, 1000, (1,)).item()
        experiment_name_with_id = experiment_name + f"_id_{uniq_id}"
        while os.path.exists(os.path.join(args.save_dir, experiment_name_with_id)):
            uniq_id += 1
            experiment_name_with_id = experiment_name + f"_id_{uniq_id}"

        experiment_name = experiment_name_with_id

    save_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save the arguments
    print(f"Saving results to {save_dir}")
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        f.write(str(args) + "\n")
    # Save the results
    with open(os.path.join(save_dir, "test_values.txt"), "w") as f:
        for iter, (cost_weight, test_loss, test_zero_one_loss) in enumerate(results):
            f.write(
                f"iter {iter},cost_weigh: {cost_weight} loss:{test_loss} zero_one_loss:{test_zero_one_loss}\n"
            )
        if len(classifiers) > 0:
            f.write("classifier weights and bias\n")
            for iter, (w, b) in enumerate(classifiers):
                f.write(f"iter {iter}, weight: {w}, bias: {b}\n")
        if len(itd_classifiers) > 0:
            f.write("itd classifier weights and bias\n")
            for iter, (w, b) in enumerate(itd_classifiers):
                f.write(f"iter {iter}, weight: {w}, bias: {b}\n")
    # Visualize the results
    visualize_datasets_and_classifiers(
        args, datasets, classifiers, itd_classifiers, experiment_name, save_dir
    )


def experiment(
    args: argparse.Namespace,
):
    if args.num_features != 2:
        args.visualize = False
    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    X, y = generate_initial_dataset(args)
    X_test, y_test = generate_initial_dataset(args, True)

    X_train, y_train, X_val, y_val = split_train_val_test(X, y, args.val_ratio)

    datasets: List[Tuple[Tensor, Tensor]] = [(X_test, y_test)]
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

    test_dataloader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    classifiers: List[Tuple[Tensor, float]] = []
    itd_classifiers: List[Tuple[Tensor, float]] = []

    cost = sml.CostNormL2(dim=1)
    cost_weight: float = args.start_cost_weight
    num_features = X.shape[1]

    itd_model = None
    itd_delta = None
    itd_cost_weight = cost_weight if args.itd_cost_override else  args.itd_start_cost_weight

    if args.non_linear:
        model = NonLinearModel(args.hidden_size)
        delta_optimizer_class = get_optimizer(args.non_linear_delta_optimizer)
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
        if args.itd:
            itd_model = NonLinearModel(args.hidden_size)
            itd_delta = sml.NonLinearStrategicDelta(
                strategic_model=model,
                cost=cost,
                cost_weight=itd_cost_weight,
                training_params=dict_training_params,
            )
    else:
        model = sml.LinearModel(num_features)
        delta = sml.LinearStrategicDelta(
            strategic_model=model, cost=cost, cost_weight=cost_weight
        )
        if args.itd:
            itd_model = sml.LinearModel(num_features)
            itd_delta = sml.LinearStrategicDelta(
                strategic_model=itd_model,
                cost=cost,
                cost_weight=itd_cost_weight,
            )

    loss_fn = get_loss_fn(args.loss_fn)
    optimizer = get_optimizer(args.optimizer)

    reg_fn: Optional[sml._LinearRegularization] = get_regularization_fn(args)

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

    if reg_fn is not None:
        reg_list: List[sml._LinearRegularization] = [reg_fn]
        model_suit.linear_regularization = reg_list

    itd_model_suit = None
    if args.itd:
        model_suit.test_delta = itd_delta
        assert itd_model is not None
        assert itd_delta is not None
        itd_model_suit = sml.ModelSuit(
            model=itd_model,
            delta=itd_delta,
            loss_fn=loss_fn,
            training_params={
                "optimizer": optimizer,
                "lr": args.lr,
            },
            # The 'in the dark' users train on the test set
            train_loader=test_dataloader,
            validation_loader=validation_dataloader,
            test_loader=test_dataloader,
        )

    results = []

    if args.pretrain_epochs > 0:
        trainer = pl.Trainer(max_epochs=args.pretrain_epochs, logger=False)
        print(f"Pre-training the model for {args.pretrain_epochs} epochs.")
        pre_train_model_suit = sml.ModelSuit(
            model=model,
            delta=sml.IdentityDelta(cost=cost, strategic_model=model),
            loss_fn=loss_fn,
            training_params={
                "optimizer": optimizer,
                "lr": args.lr,
            },
            train_loader=train_dataloader,
            validation_loader=validation_dataloader,
            test_loader=test_dataloader,
        )
        trainer.fit(pre_train_model_suit)
        if args.itd:
            assert itd_model is not None
            itd_trainer = pl.Trainer(max_epochs=args.pretrain_epochs, logger=False)
            print(
                f"Pretraining the 'in the dark' model for {args.pretrain_epochs} epochs."
            )
            pre_train_model_suit_itd = sml.ModelSuit(
                model=itd_model,
                delta=sml.IdentityDelta(cost=cost, strategic_model=itd_model),
                loss_fn=loss_fn,
                training_params={
                    "optimizer": optimizer,
                    "lr": args.lr,
                },
                train_loader=test_dataloader,
                validation_loader=validation_dataloader,
                test_loader=test_dataloader,
            )
            itd_trainer.fit(pre_train_model_suit)

    for iteration in range(args.num_iterations):
        if args.itd:
            assert itd_model_suit is not None
            itd_trainer = pl.Trainer(max_epochs=args.max_epochs, logger=False)
            print(f"model training iteration {iteration} for 'in the dark' users")
            itd_trainer.fit(itd_model_suit)

        trainer = pl.Trainer(max_epochs=args.max_epochs, logger=False)
        print(f"model training iteration {iteration}")
        trainer.fit(model_suit)
        model_suit.train_delta_for_test()
        print(f"test iteration {iteration} in the dark = {args.itd}")
        output = trainer.test(model_suit)

        print(output)
        test_value_loss = output[0]["test_loss_epoch"]
        test_value_zero_one_loss = output[0]["test_zero_one_loss_epoch"]
        results.append((cost_weight, test_value_loss, test_value_zero_one_loss))

        # Save the classifier
        if not args.non_linear:
            w, b = model.get_weight_and_bias()
            w = w.clone().view(-1)
            b = b.item()
            classifiers.append((w, b))
            if args.itd:
                assert itd_model is not None and isinstance(itd_model, sml.LinearModel)
                itd_w, itd_b = itd_model.get_weight_and_bias()
                itd_w = itd_w.clone().view(-1)
                itd_b = itd_b.item()
                itd_classifiers.append((itd_w, itd_b))

        # Create the next dataset
        if args.train_val_update_itd:
            assert itd_delta is not None
            X_train = itd_delta(X_train, y_train)
            X_val = itd_delta(X_val, y_val)
        else:
            X_train = model_suit.delta(X_train, y_train)
            X_val = model_suit.delta(X_val, y_val)

        if args.itd:
            assert itd_delta is not None
            X_test = itd_delta(X_test, y_test)
        else:
            X_test = model_suit.delta(X_test, y_test)

        if args.visualize:
            datasets.append((X_test, y_test))

        if args.model_learn_test_percentage > 0:
            # Use a percentage of the test set to train the model
            assert (
                args.model_learn_test_percentage > 0
                and args.model_learn_test_percentage < 1
            )
            num_samples = X_test.shape[0]
            num_train = int(num_samples * args.model_learn_test_percentage)
            # select random samples with their labels
            indices = torch.randperm(num_samples)[:num_train]
            X_test_train = X_test[indices]
            y_test_train = y_test[indices]
            model_suit_test_train = sml.ModelSuit(
                model=model,
                delta=sml.IdentityDelta(cost=cost, strategic_model=model),
                loss_fn=loss_fn,
                training_params={
                    "optimizer": optimizer,
                    "lr": args.lr,
                },
                train_loader=DataLoader(
                    TensorDataset(X_test_train, y_test_train),
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                ),
                validation_loader=validation_dataloader,
                test_loader=test_dataloader,
            )
            trainer_test_train = pl.Trainer(
                max_epochs=args.test_train_max_epochs, logger=False
            )
            trainer_test_train.fit(model_suit_test_train)

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
        test_dataloader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # Update the model suit
        model_suit.train_loader = train_dataloader
        model_suit.validation_loader = validation_dataloader
        model_suit.test_loader = test_dataloader

        # Update the cost weight
        cost_weight = (
            args.cost_weight_multiplier * cost_weight
        ) + args.cost_weight_addend
        model_suit.delta.set_cost_weight(cost_weight)

        if args.itd:
            assert itd_model_suit is not None
            # Update the model suit for 'in the dark' users
            itd_model_suit.train_loader = test_dataloader
            itd_model_suit.validation_loader = validation_dataloader
            itd_model_suit.test_loader = test_dataloader
            # Update the cost weight for 'in the dark' users
            if args.itd_cost_override:
                itd_cost_weight = cost_weight
            else:
                itd_cost_weight = (
                    args.itd_cost_weight_multiplier * itd_cost_weight
                ) + args.itd_cost_weight_addend

            itd_model_suit.delta.set_cost_weight(itd_cost_weight)

    output_experiment(args, datasets, classifiers, itd_classifiers, results)


def main():
    args = parse_args()
    experiment(args)


if __name__ == "__main__":
    main()
