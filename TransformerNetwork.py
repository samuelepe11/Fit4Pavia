# Import packages
import torch.nn as nn


# Class
class TransformerNetwork(nn.Module):

    def __init__(self, output_neurons=1):
        super(TransformerNetwork, self).__init__()

        # Define attributes
        self.output_neurons = output_neurons
        if output_neurons == 1:
            self.in_channels = 75
            self.num_heads = 25
            self.num_layers = 8
            self.hidden_dim = 2048
        else:
            print("TODO")

        # Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.in_channels, nhead=self.num_heads, batch_first=True,
                                                    dim_feedforward=self.hidden_dim, norm_first=True)
        self.trans = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(in_features=self.in_channels, out_features=self.output_neurons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.trans(x)
        out = out.mean(1)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out
