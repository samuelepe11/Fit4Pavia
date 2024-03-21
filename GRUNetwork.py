# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class GRUNetwork(RNNetwork):

    def __init__(self, output_neurons=1):
        super(GRUNetwork, self).__init__(output_neurons)

        # Define attributes
        self.output_neurons = output_neurons
        if output_neurons == 1:
            self.hidden_dim = 64
        else:
            print("TODO")

        # Adjust layers
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_neurons)
