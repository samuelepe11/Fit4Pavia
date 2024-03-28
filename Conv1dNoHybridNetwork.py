# Import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Masking, Conv1D, MaxPooling1D, Flatten, Dense


# Class
class Conv1dNoHybridNetwork:

    # Define class attributes
    mask_value = 0.0

    def __init__(self, output_neurons=1, conv2_flag=False):

        # Define attributes
        self.output_neurons = output_neurons
        if output_neurons == 1:
            self.layer_dims = [16, 32]
            self.hidden_dim = 32
        else:
            print("TODO")

        # Layers
        self.conv2_flag = conv2_flag
        self.mask = Masking(mask_value=self.mask_value)
        self.flat = Flatten()
        if self.output_neurons == 1:
            activation = "sigmoid"
        else:
            activation = "softmax"
        self.dense = Dense(self.output_neurons, activation=activation)

        self.model = Sequential()
        self.model.add(self.mask)
        for i in range(len(self.layer_dims)):
            self.model.add(Conv1D(filters=self.layer_dims[i], kernel_size=3, activation="relu"))
            self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(self.flat)
        self.model.add(self.dense)

    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    def train(self, x, y, epochs, batch_size):
        if self.conv2_flag:
            x = np.expand_dims(x, 3)

        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return history

    def predict(self, x):
        # Handle previous versions of the Conv1dNoHybridNetwork class (no conv2_flag attribute)
        if "conv2_flag" not in self.__dict__.keys():
            self.conv2_flag = False

        if self.conv2_flag:
            x = np.expand_dims(x, 3)

        return self.model.predict(x, verbose=0)

    def evaluate(self, x, y):
        if self.conv2_flag:
            x = np.expand_dims(x, 3)

        return self.model.evaluate(x, y, verbose=0)
