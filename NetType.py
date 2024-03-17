# Import packages
from enum import Enum


# Class
class NetType(Enum):
    RNN = "RNN"
    LSTM = "LSTM"
    BLSTM = "Bi-LSTM"
    GRU = "GRU"
    CONV1D = "Convolutional 1D"
    CONV2D = "Convolutional 2D"
    CONV1D_NO_HYBRID = "Convolutional 1D without RNN layers"
    CONV2D_NO_HYBRID = "Convolutional 2D without RNN layers"
    TCN = "Temporal Convolutional Network"
