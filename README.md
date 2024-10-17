# Iterative Strategic Classification Experiment

## Introduction

This experiment demonstrates iterative strategic classification using the `strategic_ml` library. In strategic classification, agents (users) may manipulate their features to receive favorable outcomes from predictive models. This experiment simulates multiple iterations where both the classifier and agents adapt to each other over time.

At each iteration:

- **Classifier Training**: The classifier is trained on the current dataset.
- **Strategic Manipulation**: Agents adjust their features strategically to improve their outcomes, considering the cost of manipulation.
- **Dataset Update**: The dataset is updated with the agents' new features.
- **Cost Update**: The cost of manipulation may increase over iterations, simulating increasing difficulty or expense for agents to manipulate their features.
- **Optional "In The Dark" (ITD) Scenario**: Agents may be "in the dark," meaning they do not know the classifier's parameters and must adapt differently.

This experiment allows for various configurations, including different datasets, models (linear or non-linear), cost functions, and strategic behaviors.

## Prerequisites

- **Python**: Version 3.6 or higher
- **Packages**:
  - `torch`
  - `torchvision`
  - `pytorch-lightning`
  - `scikit-learn`
  - `matplotlib`
  - `numpy`
  - `argparse`
- **`strategic_ml` Library**: Ensure that the `strategic_ml` library is installed and accessible.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/strategic_ml_experiment.git
   cd strategic_ml_experiment
   ```

2. **Install Required Packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

To run the experiment, use the following command:

```bash
python iterative_strategic_ml.py [arguments]
```

### Example

```bash
python iterative_strategic_ml.py --dataset gaussian --num_iterations 10 --visualize
```

This command runs the experiment using the Gaussian dataset for 10 iterations with visualization enabled.

## Experiment Overview

### Main Steps

1. **Dataset Generation**: An initial dataset is generated based on the specified parameters (e.g., dataset type, noise level, number of samples).

2. **Model Initialization**: A predictive model is initialized. You can choose between a linear model and a non-linear model with hidden layers.

3. **Strategic Delta Definition**: A strategic delta represents how agents adjust their features to improve outcomes while considering manipulation costs.

4. **Training Loop**: The experiment runs for a specified number of iterations (`--num_iterations`). In each iteration:
   - The model is trained on the current dataset.
   - Agents adjust their features using the strategic delta.
   - The dataset is updated with the new features.
   - The cost weight is updated for the next iteration.

5. **Optional ITD Scenario**: In the "In The Dark" scenario, agents do not know the classifier's parameters and must adapt differently.

6. **Visualization**: If enabled, the experiment visualizes the datasets and classifiers over iterations.

### Visualization

The visualization includes:

- Data points with positive and negative labels.
- Movements of data points due to agents' strategic manipulations.
- Decision boundaries of the classifier over iterations.
- Optional ITD classifiers' decision boundaries.

## Command-Line Arguments

You can customize the experiment using various command-line arguments.

### Dataset Parameters

- `--num_samples`: Number of training samples (default: `1000`).
- `--added_noise`: Noise level added to the dataset features (default: `0.0`).
- `--test_samples`: Number of test samples (default: `1000`).
- `--test_added_noise`: Noise level added to the test dataset features (default: `0.0`).
- `--num_features`: Number of features (default: `2`). Only used for the linear dataset.
- `--dataset`: Dataset to use (`linear`, `gaussian`, `circular`, `spiral`, `moons`). Default: `gaussian`.
- `--mean_pos`: Mean of the positive class for the Gaussian dataset (default: `2.0`).
- `--mean_neg`: Mean of the negative class for the Gaussian dataset (default: `-2.0`).
- `--std_dev`: Standard deviation for the Gaussian clusters (default: `1.0`).
- `--data_radius_multiplier`: Radius multiplier for the circular dataset (default: `1.0`).
- `--label_radius`: Radius that indicates where the label is positive in the circular dataset.
- `--noise`: Noise level for the spiral and moons datasets (default: `0.1`).
- `--val_ratio`: Validation set ratio (default: `0.2`).

### Training Parameters

- `--max_epochs`: Maximum number of epochs for training (default: `100`).
- `--batch_size`: Batch size (default: `32`).
- `--lr`: Learning rate (default: `0.1`).
- `--num_workers`: Number of workers for data loaders (default: `0`).

### Strategic Learning Parameters

- `--start_cost_weight`: Initial cost weight for agents' manipulation cost (default: `1.0`).
- `--cost_weight_multiplier`: Multiplier to increase the cost weight after each iteration (default: `1.5`).
- `--cost_weight_addend`: Addend to increase the cost weight after each iteration (default: `0.0`).
- `--loss_fn`: Loss function to use (`bce`, `mse`, `hinge`). Default: `bce`.
- `--optimizer`: Optimizer to use (`sgd`, `adam`, `adagrad`). Default: `adam`.
- `--linear_regulation_fn`: Linear regularization function (`l1`, `l2`, `elastic`). Default: `l1`.
- `--linear_regulation_strength`: Strength of the linear regularization (default: `0.01`).
- `--elastic_ratio`: Ratio for elastic net regularization (default: `0.5`).

### Non-Linear Model Parameters

- `--non_linear`: Use a non-linear model (default: `False`).
- `--hidden_size`: Hidden layer size for the non-linear model (default: `2`).
- `--non_linear_delta_optimizer`: Optimizer for the non-linear delta (`sgd`, `adam`, `adagrad`). Default: `adam`.
- `--non_linear_delta_lr`: Learning rate for the non-linear delta (default: `0.001`).
- `--non_linear_delta_max_epochs`: Maximum epochs for training the non-linear delta (default: `65`).
- `--non_linear_delta_temp`: Temperature parameter for the non-linear delta (default: `27.0`).

### "In The Dark" (ITD) Parameters

- `--itd`: Use the "In The Dark" scenario (default: `False`).
- `--itd_start_cost_weight`: Initial cost weight for ITD agents (default: `1.0`).
- `--itd_cost_weight_multiplier`: Cost weight multiplier for ITD agents (default: `1.5`).
- `--itd_cost_weight_addend`: Cost weight addend for ITD agents (default: `0.0`).
- `--train_val_update_itd`: Update training and validation data using the ITD delta (default: `False`).
- `--model_learn_test_percentage`: Percentage of the test set to use for training the model after each iteration (default: `0.0`).
- `--test_train_max_epochs`: Maximum epochs for training the model on the test set (default: `20`).

### Other Parameters

- `--seed`: Random seed (default: `0`).
- `--num_iterations`: Number of iterations to run (default: `10`).
- `--save_dir`: Directory to save the results (default: `results`).
- `--plot_name`: Name of the plot file.
- `--plot_fraction`: Fraction of data points to plot (default: `0.5`).
- `--visualize`: Enable visualization of results (default: `True`).
- `--add_uniq_it`: Adds a uniq id to the save dir (defalue: `False`).

## Results and Visualization

After running the experiment, the results are saved in the specified `--save_dir`. The following files are generated:

- **`args.txt`**: Contains the command-line arguments used for the experiment.
- **`val_values.txt`**: Logs the validation loss and zero-one loss at each iteration.
- **Visualization Plots**: If visualization is enabled, plots showing the datasets and classifiers over iterations are saved.

### Understanding the Visualization

- **Data Points**:
  - **Positive Labels**: Represented in blue shades.
  - **Negative Labels**: Represented in red shades.
- **Data Movement**: Arrows indicate how data points (agents) adjust their features over iterations.
- **Decision Boundaries**:
  - **Classifier**: Shown in green shades, representing the classifier's decision boundary over iterations.
  - **ITD Classifier**: If ITD is enabled, shown in purple shades.
- **Cost Parameters**: Displayed in the title for reference.

## Understanding the Experiment

This experiment simulates a dynamic interaction between agents and the classifier:

- **Agents**:
  - Adjust their features strategically to receive favorable predictions.
  - Consider the cost of manipulation, which may increase over iterations.
- **Classifier**:
  - Updates its model based on the current dataset, which includes agents' manipulated features.
  - May employ regularization to discourage overfitting or promote certain behaviors.
- **Iterations**:
  - Represent time steps where both agents and the classifier adapt to each other.
  - Reflect how strategic behavior evolves over time.

By increasing the cost weight over iterations, the experiment models scenarios where it becomes increasingly costly for agents to manipulate their features, leading to different strategic behaviors.

The "In The Dark" scenario models agents who do not know the classifier's parameters, simulating a more realistic situation where agents cannot perfectly predict the classifier's responses.

## Customization and Extensions

You can customize the experiment by:

- **Changing Datasets**: Use different datasets (`linear`, `gaussian`, `circular`, `spiral`, `moons`) to observe how dynamics change with different data distributions.
- **Adjusting Cost Parameters**: Modify the cost weight, multiplier, and addend to simulate different manipulation cost dynamics.
- **Using Non-Linear Models**: Enable the `--non_linear` option to use a non-linear model, which may capture more complex patterns.
- **Experimenting with ITD**: Enable the `--itd` option to include agents who are "in the dark" about the classifier's parameters.
- **Implementing Custom Strategies**: Modify the strategic delta or cost functions to implement custom agent behaviors.

## Troubleshooting

- **Missing Packages**: Ensure all required packages are installed. Use the provided `requirements.txt`.
- **Memory Issues**: Reduce the number of samples (`--num_samples`, `--test_samples`) or batch size (`--batch_size`) if you encounter memory errors.
- **Visualization Errors**: Ensure that `matplotlib` is installed and your environment supports plotting (e.g., use `%matplotlib inline` in Jupyter notebooks).
- **Randomness**: Set the random seed (`--seed`) for reproducible results.

## Acknowledgments

This experiment utilizes the `strategic_ml` library to model strategic classification scenarios. The library provides tools for:

- Modeling strategic behavior through strategic deltas.
- Defining cost functions for feature manipulation.
- Implementing regularization techniques.
- Training and evaluating models in strategic settings.

For more details on the `strategic_ml` library, refer to its [documentation](https://github.com/orimintz21/strategic_ml).

---

**Note**: This README is specific to the iterative strategic classification experiment and aims to guide users in understanding, running, and customizing the experiment. For details on the underlying library, please refer to the `strategic_ml` documentation.
