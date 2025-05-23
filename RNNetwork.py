# Import packages
import torch.nn as nn


# Class
class RNNetwork(nn.Module):

    def __init__(self, num_classes=2, is_rehab=False):
        super(RNNetwork, self).__init__()

        # Define attributes
        self.num_classes = num_classes
        self.in_channels = 75
        if num_classes == 2:
            self.output_neurons = 1
            if not is_rehab:
                self.hidden_dim = 32
                self.num_layers = 1
            else:
                # TODO
                self.hidden_dim = 32
                self.num_layers = 1
        else:
            if not is_rehab:
                # TODO
                self.hidden_dim = 32
                self.num_layers = 1
            else:
                # TODO
                self.in_channels = 26
                self.output_neurons = num_classes
                self.hidden_dim = 32
                self.num_layers = 1

        # Layers
        self.rnn = nn.RNN(input_size=self.in_channels, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_neurons)
        if self.output_neurons == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x, avoid_eval=False):
        try:
            out, _ = self.rnn(x)
        except AttributeError:
            # Handle previous versions of the LSTMNetwork class (no rnn attribute)
            out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out
