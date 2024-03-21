# Import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class Conv2dNoHybridNetwork(Conv1dNoHybridNetwork):

    def __init__(self, output_neurons=1):
        super().__init__(output_neurons)

        # Define attributes
        self.conv2_flag = True
        self.output_neurons = output_neurons
        if output_neurons == 1:
            self.num_conv_layers = 3
        else:
            print("TODO")

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        for i in range(self.num_conv_layers):
            self.model.add(Conv2D(filters=1, kernel_size=3, activation="relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(self.flat)
        self.model.add(self.dense)
