import torch.nn as nn

class DynamicClassifier(nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_layers_config):
        super(DynamicClassifier, self).__init__()
        layers = []
        current_input_dim = input_dimension

        for layer_size in hidden_layers_config:
            if layer_size > 0:
                layers.append(nn.Linear(current_input_dim, layer_size))
                layers.append(nn.ReLU())
                current_input_dim = layer_size

        layers.append(nn.Linear(current_input_dim, output_dimension))
        self.network_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.network_layers(x)
