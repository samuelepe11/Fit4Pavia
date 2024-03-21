# Import packages
import torch.nn as nn

from Conv1dNetwork import Conv1dNetwork


# Class
class Conv2dNetwork(Conv1dNetwork):

    def __init__(self, output_neurons=1):
        super(Conv2dNetwork, self).__init__(output_neurons)

        # Define attributes
        if self.output_neurons == 1:
            self.num_conv_layers = 2
            self.hidden_dim = 64
        else:
            print("TODO")

        # Layers
        new_channels = self.in_channels
        for i in range(self.num_conv_layers):
            self.__dict__["conv" + str(i)] = nn.Conv2d(1, 1, kernel_size=3, stride=1)
            new_channels -= 2
            self.__dict__["batch_norm" + str(i)] = nn.BatchNorm1d(new_channels)

        self.rnn = nn.RNN(input_size=new_channels, hidden_size=self.hidden_dim, num_layers=self.num_rnn_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, out_features=self.output_neurons)
