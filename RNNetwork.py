# Import packages
import torch.nn as nn


# Class
class RNNetwork(nn.Module):

    # Define class attributes
    in_channels = 75
    hidden_dim = 32
    num_layers = 1

    def __init__(self):
        super(RNNetwork, self).__init__()

        # Layers
        self.rnn = nn.RNN(input_size=self.in_channels, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)
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
