# Import packages
import torch.nn as nn


# Class
class RNNetwork(nn.Module):

    def __init__(self, output_neurons=1):
        super(RNNetwork, self).__init__()

        # Define attributes
        self.output_neurons = output_neurons
        if output_neurons == 1:
            self.in_channels = 75
            self.hidden_dim = 32
            self.num_layers = 1
        else:
            print("TODO")

        # Layers
        self.rnn = nn.RNN(input_size=self.in_channels, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_neurons)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            out, _ = self.rnn(x)
        except AttributeError:
            # Handle previous versions of the LSTMNetwork class (no rnn attribute)
            out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out
