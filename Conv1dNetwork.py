# Import packages
import torch
import torch.nn as nn


# Class
class Conv1dNetwork(nn.Module):

    def __init__(self, num_classes=2, is_2d=False, binary_output=False):
        super(Conv1dNetwork, self).__init__()
        self.is_2d = is_2d

        # Define attributes
        self.in_channels = 75
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.layer_dims = [self.in_channels, 16, 32, 64]
            self.hidden_dim = 128
            self.num_rnn_layers = 1
            if not binary_output:
                self.output_neurons = 1
            else:
                self.output_neurons = 2
        else:
            self.layer_dims = [self.in_channels, 128, 128, 128]
            self.hidden_dim = 128
            self.num_rnn_layers = 1
            self.output_neurons = num_classes
        self.num_conv_layers = len(self.layer_dims) - 1

        # Layers
        for i in range(self.num_conv_layers):
            self.__dict__["conv" + str(i)] = nn.Conv1d(self.layer_dims[i], self.layer_dims[i + 1], kernel_size=3,
                                                       stride=1)
            self.__dict__["conv" + str(i) + "pad"] = nn.Conv1d(self.layer_dims[i], self.layer_dims[i + 1],
                                                               kernel_size=3, stride=1, padding=1)
            self.__dict__["pool" + str(i)] = nn.MaxPool1d(kernel_size=2)
            self.__dict__["relu" + str(i)] = nn.ReLU()
            self.__dict__["batch_norm" + str(i)] = nn.BatchNorm1d(self.layer_dims[i + 1])

        if self.num_classes == 2:
            self.rnn = nn.RNN(input_size=self.layer_dims[-1], hidden_size=self.hidden_dim, num_layers=self.num_rnn_layers,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=self.layer_dims[-1], hidden_size=self.hidden_dim, num_layers=self.num_rnn_layers,
                               batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, out_features=self.output_neurons)

        if self.output_neurons == 1:
            sigmoid = nn.Sigmoid()
        else:
            sigmoid = nn.Softmax(dim=1)
        self.sigmoid = sigmoid

        # CAM attributes
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, layer_interrupt=None):
        # Apply network
        out = x.permute(0, 2, 1)

        if self.is_2d:
            out = out.unsqueeze(0)

        # Handle previous versions of the Conv1dNetwork class (no num_conv_layers attribute)
        if "num_conv_layers" not in self.__dict__.keys():
            self.num_conv_layers = len([x for x in self.__dict__.keys() if x.startswith("conv")])

        target_activation = None
        for i in range(self.num_conv_layers):
            conv_layer = "conv" + str(i)
            if out.shape[-1] < 3:
                conv_layer += "pad"
            out = self.__dict__[conv_layer](out)
            if layer_interrupt == "conv" + str(i):
                target_activation = out
                h = out.register_hook(self.activations_hook)

                if self.num_classes == 2:
                    out = self.__dict__["pool" + str(i)](out)
                out = self.__dict__["relu" + str(i)](out)
                out = self.__dict__["batch_norm" + str(i)](out)

        if self.is_2d:
            out = torch.mean(out, dim=2)

        out = out.permute(0, 2, 1)
        out, _ = self.rnn(out)
        out = self.fc(out[:, -1, :])

        if layer_interrupt is not None:
            return out, target_activation

        out = self.sigmoid(out)
        return out
