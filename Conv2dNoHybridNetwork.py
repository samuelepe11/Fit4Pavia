# Import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class Conv2dNoHybridNetwork(Conv1dNoHybridNetwork):

    # Define class attributes
    layer_dims = [16, 32]
    hidden_dim = 32

    def __init__(self):
        super().__init__()

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        for i in range(self.num_conv_layers):
            self.model.add(Conv2D(filters=self.layer_dims[i], kernel_size=3, activation="relu"))
            self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(self.flat)
        self.model.add(self.dense)
