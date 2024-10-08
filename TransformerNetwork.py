# Import packages
import torch.nn as nn


# Class
class TransformerNetwork(nn.Module):

    def __init__(self, num_classes=2, is_rehab=False):
        super(TransformerNetwork, self).__init__()

        # Define attributes
        self.num_classes = num_classes
        if num_classes == 2:
            if not is_rehab:
                self.in_channels = 75
                self.num_heads = 25
                self.num_layers = 8
                self.hidden_dim = 2048
                self.output_neurons = 1
            else:
                print("TODO")
        else:
            if not is_rehab:
                print("TODO")
            else:
                print("TODO")

        # Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.in_channels, nhead=self.num_heads, batch_first=True,
                                                    dim_feedforward=self.hidden_dim, norm_first=True)
        self.trans = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers, enable_nested_tensor=False)
        self.fc = nn.Linear(in_features=self.in_channels, out_features=self.output_neurons)
        if self.output_neurons == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.trans(x)
        out = out.mean(1)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out
