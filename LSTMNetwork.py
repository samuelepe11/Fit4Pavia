# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class LSTMNetwork(RNNetwork):

    def __init__(self, bidirectional, num_classes=2, is_rehab=False):
        super(LSTMNetwork, self).__init__(num_classes=num_classes, is_rehab=is_rehab)

        # Define attributes
        self.bidirectional = bidirectional
        if num_classes == 2:
            if not is_rehab:
                self.num_layers = 1
                self.hidden_dim = 32
                self.lr = 0.01
            else:
                self.num_layers = 1
                self.hidden_dim = 32
                self.lr = 0.01
        else:
            if not is_rehab:
                self.num_layers = 1
                self.hidden_dim = 128
                self.lr = 0.001
            else:
                self.num_layers = 1
                self.hidden_dim = 256
                self.lr = 0.001 if not bidirectional else 0.00001

        self.fc_input = self.hidden_dim
        if self.bidirectional:
            self.fc_input *= 2

        # Adjust layers
        self.rnn = nn.LSTM(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True,
                           bidirectional=bidirectional, num_layers=self.num_layers)
        self.fc = nn.Linear(in_features=self.fc_input, out_features=self.output_neurons)
