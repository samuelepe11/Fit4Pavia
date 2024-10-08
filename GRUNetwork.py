# Import packages
import torch.nn as nn
from RNNetwork import RNNetwork


# Class
class GRUNetwork(RNNetwork):

    def __init__(self, num_classes=2, is_rehab=False):
        super(GRUNetwork, self).__init__(num_classes=num_classes, is_rehab=is_rehab)

        # Define attributes
        if num_classes == 2:
            if not is_rehab:
                self.hidden_dim = 64
            else:
                print("TODO")
        else:
            if not is_rehab:
                print("TODO")
            else:
                print("TODO")

        # Adjust layers
        self.rnn = nn.GRU(input_size=self.in_channels, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.output_neurons)
