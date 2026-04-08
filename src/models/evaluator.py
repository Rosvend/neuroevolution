import torch
import torch.nn as nn
import torch.optim as optim

class ModelEvaluator:
    def __init__(self, input_dim=784, output_dim=24, epochs=5, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def train_single_epoch(self, model, train_loader, optimizer):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
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

    def evaluate(self, chromosome, train_loader, val_loader):
        from src.models.dynamic_model import DynamicClassifier
        model = DynamicClassifier(self.input_dim, self.output_dim, chromosome).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        best_accuracy = 0.0
        for _ in range(self.epochs):
            self.train_single_epoch(model, train_loader, optimizer)
            accuracy = self.validate(model, val_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        return best_accuracy
