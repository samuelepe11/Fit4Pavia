# Import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class Conv2dNoHybridNetwork(Conv1dNoHybridNetwork):

    def __init__(self, num_classes=2, binary_output=False):
        super().__init__(num_classes, True, binary_output)

        # Define attributes
        if num_classes == 2:
            self.layer_dims = [16, 32]
            self.hidden_dim = 32
        else:
            # TODO
            self.layer_dims = [16, 32]
            self.hidden_dim = 32

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        for i in range(len(self.layer_dims)):
            self.model.add(Conv2D(filters=self.layer_dims[i], kernel_size=3, activation="relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(GlobalAveragePooling2D(data_format="channels_last", keepdims=False))
        self.model.add(self.dense)
