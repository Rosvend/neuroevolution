import torch.nn as nn

class DynamicClassifier(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        hidden_layers_config,
        dropout_rate=0.0,
        use_batch_norm=False,
        activation_name="relu",
    ):
        super(DynamicClassifier, self).__init__()
        layers = []
        current_input_dim = input_dimension

        activation_layer = self._get_activation_layer(activation_name)

        for layer_size in hidden_layers_config:
            if layer_size > 0:
                layers.append(nn.Linear(current_input_dim, layer_size))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_size))
                layers.append(activation_layer())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))
                current_input_dim = layer_size

        layers.append(nn.Linear(current_input_dim, output_dimension))
        self.network_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.network_layers(x)

    @staticmethod
    def _get_activation_layer(activation_name):
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        return activation_map.get(activation_name, nn.ReLU)
