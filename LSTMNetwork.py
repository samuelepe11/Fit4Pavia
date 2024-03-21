# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class LSTMNetwork(RNNetwork):

    def __init__(self, bidirectional, output_neurons=1):
        super(LSTMNetwork, self).__init__(output_neurons)

        # Define attributes
        self.output_neurons = output_neurons
        self.bidirectional = bidirectional
        if output_neurons == 1:
            self.hidden_dim = 32
        else:
            print("TODO")

        self.fc_input = self.hidden_dim
        if self.bidirectional:
            self.fc_input *= 2

        # Adjust layers
        self.rnn = nn.LSTM(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=self.fc_input, out_features=self.output_neurons)
