import optuna

class BayesianOptimizer:
    def __init__(self, total_trials=100, maximum_layers=6, minimum_neurons=32, maximum_neurons=1024):
        self.total_trials = total_trials
        self.maximum_layers = maximum_layers
        self.minimum_neurons = minimum_neurons
        self.maximum_neurons = maximum_neurons
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
        )

    def build_objective_function(self, evaluator, train_loader, validation_loader):
        def objective(trial):
            layer_count = trial.suggest_int("layer_count", 1, self.maximum_layers)
            layer_configuration = []

            current_max_neurons = self.maximum_neurons
            for layer_index in range(layer_count):
                neurons = trial.suggest_int(
                    f"neurons_layer_{layer_index}",
                    self.minimum_neurons,
                    current_max_neurons,
                    log=True,
                )
                layer_configuration.append(neurons)
                current_max_neurons = neurons

            layer_configuration.extend([0] * (self.maximum_layers - layer_count))

            model_config = {
                "hidden_layers_config": layer_configuration,
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.55),
                "use_batch_norm": trial.suggest_categorical("use_batch_norm", [True, False]),
                "activation_name": trial.suggest_categorical("activation_name", ["relu", "gelu", "leaky_relu"]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "optimizer_name": trial.suggest_categorical("optimizer_name", ["adamw", "rmsprop"]),
                "gradient_clip_norm": trial.suggest_float("gradient_clip_norm", 0.5, 5.0),
                "epochs": trial.suggest_int("epochs", 8, 24),
            }

            return evaluator.evaluate(model_config, train_loader, validation_loader, trial=trial)
        
        return objective

    def execute_optimization(self, evaluator, train_loader, validation_loader):
        objective_function = self.build_objective_function(evaluator, train_loader, validation_loader)
        self.study.optimize(objective_function, n_trials=self.total_trials)

    def extract_best_results(self):
        best_trial_data = self.study.best_trial
        return best_trial_data.value, best_trial_data.params
