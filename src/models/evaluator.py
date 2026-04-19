import torch
import torch.nn as nn
import torch.optim as optim
import optuna

class ModelEvaluator:
    def __init__(self, input_dim=784, output_dim=24, epochs=12, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def train_single_epoch(self, model, train_loader, optimizer, gradient_clip_norm=None):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

    def validate(self, model, val_loader):
        model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        return correct_predictions / total_samples

    def evaluate(self, model_config, train_loader, val_loader, trial=None):
        from src.models.dynamic_model import DynamicClassifier

        model = DynamicClassifier(
            self.input_dim,
            self.output_dim,
            model_config["hidden_layers_config"],
            dropout_rate=model_config.get("dropout_rate", 0.0),
            use_batch_norm=model_config.get("use_batch_norm", False),
            activation_name=model_config.get("activation_name", "relu"),
        ).to(self.device)

        learning_rate = model_config.get("learning_rate", self.learning_rate)
        weight_decay = model_config.get("weight_decay", 0.0)
        optimizer_name = model_config.get("optimizer_name", "adamw")
        max_epochs = model_config.get("epochs", self.epochs)
        gradient_clip_norm = model_config.get("gradient_clip_norm")

        if optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )

        best_accuracy = 0.0
        epochs_without_improvement = 0
        early_stopping_patience = 4

        for epoch in range(max_epochs):
            self.train_single_epoch(model, train_loader, optimizer, gradient_clip_norm=gradient_clip_norm)
            accuracy = self.validate(model, val_loader)
            scheduler.step(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if trial is not None:
                trial.report(accuracy, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if epochs_without_improvement >= early_stopping_patience:
                break

        return best_accuracy
