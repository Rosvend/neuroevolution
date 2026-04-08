# Sign Language Neuroevolution

This project aims to optimize the architecture of a deep neural network using genetic algorithms to classify sign language alphabet images from the Sign Language MNIST dataset.

## Objective
Finding the optimal number of hidden layers and neurons per layer through a population-based evolutionary optimization process.

## Project Structure
The project follows clean architecture principles, separating logic into domain-specific directories:

- **src/data/**: Contains dataset definitions and data loaders for Sign Language MNIST.
- **src/models/**: Defines the dynamic neural network architecture and model evaluation logic.
- **src/optimization/**: Implements the genetic algorithm, including selection, crossover, and mutation.
- **src/utils/**: Provides visualization tools for tracking fitness evolution.
- **src/main.py**: Orchestrates the experiment and runs the optimization loop.

## Setup and Requirements
### Dependencies
- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Dataset
The dataset must be placed in the following directory:
`datamunge/sign-language-mnist/`

Required files:
- `sign_mnist_train.csv`
- `sign_mnist_test.csv`

## Execution
To start the evolutionary optimization process, run:
```bash
python src/main.py
```
The script will output the best fitness (accuracy) achieved in each generation and generate a visualization of the evolution progress.
