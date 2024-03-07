# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class LSTMNetwork(RNNetwork):

    hidden_dim = 32

    def __init__(self, bidirectional):
        super(LSTMNetwork, self).__init__()

        # Adjust layers
        self.rnn = nn.LSTM(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=self.hidden_dim * 2, out_features=1)
