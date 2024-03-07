# Import packages
import torch.nn as nn


# Class
class Conv1dNetwork(nn.Module):

    # Define class attributes
    in_channels = 75
    layer_dims = [in_channels, 16, 32, 64]
    num_conv_layers = len(layer_dims) - 1
    hidden_dim = 128
    num_rnn_layers = 1

    def __init__(self):
        super(Conv1dNetwork, self).__init__()

        # Layers
        for i in range(self.num_conv_layers):
            self.__dict__["conv" + str(i)] = nn.Conv1d(self.layer_dims[i], self.layer_dims[i + 1], kernel_size=3,
                                                       stride=1)
            self.__dict__["pool" + str(i)] = nn.MaxPool1d(kernel_size=2)
            self.__dict__["relu" + str(i)] = nn.ReLU()
            self.__dict__["batch_norm" + str(i)] = nn.BatchNorm1d(self.layer_dims[i + 1])

        self.rnn = nn.RNN(input_size=self.layer_dims[-1], hidden_size=self.hidden_dim, num_layers=self.num_rnn_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply network
        out = x.permute(0, 2, 1)
        for i in range(self.num_conv_layers):
            out = self.__dict__["conv" + str(i)](out)
            out = self.__dict__["pool" + str(i)](out)
            out = self.__dict__["relu" + str(i)](out)
            out = self.__dict__["batch_norm" + str(i)](out)

        out = out.permute(0, 2, 1)
        out, _ = self.rnn(out)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out
