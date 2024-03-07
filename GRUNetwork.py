# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class GRUNetwork(RNNetwork):

    hidden_dim = 64

    def __init__(self):
        super(GRUNetwork, self).__init__()

        # Adjust layers
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)
