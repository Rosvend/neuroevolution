import optuna

class BayesianOptimizer:
    def __init__(self, total_trials=50, maximum_layers=5, minimum_neurons=16, maximum_neurons=512):
        self.total_trials = total_trials
        self.maximum_layers = maximum_layers
        self.minimum_neurons = minimum_neurons
        self.maximum_neurons = maximum_neurons
        self.study = optuna.create_study(direction="maximize")

    def build_objective_function(self, evaluator, train_loader, validation_loader):
        def objective(trial):
            layer_count = trial.suggest_int("layer_count", 1, self.maximum_layers)
            layer_configuration = []
            
            for layer_index in range(layer_count):
                layer_configuration.append(trial.suggest_int(f"neurons_layer_{layer_index}", self.minimum_neurons, self.maximum_neurons))
            
            layer_configuration.extend([0] * (self.maximum_layers - layer_count))
            
            return evaluator.evaluate(layer_configuration, train_loader, validation_loader)
        
        return objective

    def execute_optimization(self, evaluator, train_loader, validation_loader):
        objective_function = self.build_objective_function(evaluator, train_loader, validation_loader)
        self.study.optimize(objective_function, n_trials=self.total_trials)

    def extract_best_results(self):
        best_trial_data = self.study.best_trial
        return best_trial_data.value, best_trial_data.params
