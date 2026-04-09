# Sign Language MNIST Architecture Optimization

This project automates the search for an optimal deep neural network architecture to classify sign language alphabet images from the Sign Language MNIST dataset.

## Objective
Finding the optimal number of hidden layers and neurons per layer through Bayesian Optimization using Optuna. The search space is explored efficiently to maximize classification accuracy on the validation set.

## Project Structure
The project follows clean architecture principles, separating logic into domain-specific directories:

- **src/data/**: Handles dataset downloading, CSV parsing, and PyTorch data loading.
- **src/models/**: Defines the dynamic neural network architecture and the model evaluation strategy.
- **src/optimization/**: Implements the Bayesian Optimization logic using the Optuna framework.
- **src/utils/**: Provides visualization tools for tracking the optimization progress across trials.
- **src/main.py**: Orchestrates the experiment and executes the optimization trials.

## Setup and Requirements
### Package Management
This project uses **uv** for dependency management and environment reproduction.

### Dependencies
- Python 3.10+
- PyTorch
- Optuna
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle API

### Dataset
The dataset is automatically downloaded from Kaggle if not present. Ensure your Kaggle API credentials are configured in your system.

## Execution
To install dependencies and synchronize the environment:
```bash
uv sync
```

To run the optimization trials (default 100 trials):
```bash
uv run python src/main.py
```
The script will display the results for each trial, the final best architecture found, and a visualization of the accuracy evolution.