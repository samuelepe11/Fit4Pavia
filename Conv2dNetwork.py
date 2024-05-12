# Import packages
import torch.nn as nn

from Conv1dNetwork import Conv1dNetwork


# Class
class Conv2dNetwork(Conv1dNetwork):

    def __init__(self, num_classes=2, binary_output=False):
        super(Conv2dNetwork, self).__init__(num_classes, True, binary_output)

        # Define attributes
        if self.num_classes == 2:
            self.layer_dims = [1, 16, 32, 64]
            self.hidden_dim = 64
            self.num_rnn_layers = 1
        else:
            self.layer_dims = [1, 64, 64, 64]
            self.hidden_dim = 64
            self.num_rnn_layers = 1
        self.num_conv_layers = len(self.layer_dims) - 1

        # Layers
        for i in range(self.num_conv_layers):
            self.__dict__["conv" + str(i)] = nn.Conv2d(self.layer_dims[i], self.layer_dims[i + 1], kernel_size=3,
                                                       stride=1)
            self.__dict__["conv" + str(i) + "pad"] = nn.Conv2d(self.layer_dims[i], self.layer_dims[i + 1],
                                                               kernel_size=3, stride=1, padding=1)
            self.__dict__["pool" + str(i)] = nn.MaxPool2d(kernel_size=(2, 2))
            self.__dict__["batch_norm" + str(i)] = nn.BatchNorm2d(self.layer_dims[i + 1])
            self.__dict__["dropout" + str(i)] = nn.Dropout2d(p=0.1)

        if self.num_classes == 2:
            self.rnn = nn.RNN(input_size=self.layer_dims[-1], hidden_size=self.hidden_dim,
                              num_layers=self.num_rnn_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=self.layer_dims[-1], hidden_size=self.hidden_dim,
                               num_layers=self.num_rnn_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, out_features=self.output_neurons)
