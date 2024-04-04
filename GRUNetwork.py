# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class GRUNetwork(RNNetwork):

    def __init__(self, num_classes=2):
        super(GRUNetwork, self).__init__(num_classes)

        # Define attributes
        if num_classes == 2:
            self.hidden_dim = 64
        else:
            # TODO
            self.hidden_dim = 64

        # Adjust layers
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_neurons)
