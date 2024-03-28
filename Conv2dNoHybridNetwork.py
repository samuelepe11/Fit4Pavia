# Import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class Conv2dNoHybridNetwork(Conv1dNoHybridNetwork):

    def __init__(self, output_neurons=1):
        super().__init__(output_neurons, True)

        # Define attributes
        self.output_neurons = output_neurons
        if output_neurons == 1:
            self.layer_dims = [16, 32]
        else:
            print("TODO")

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        for i in range(len(self.layer_dims)):
            self.model.add(Conv2D(filters=self.layer_dims[i], kernel_size=3, activation="relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(GlobalAveragePooling2D(data_format="channels_last", keepdims=False))
        self.model.add(self.dense)
