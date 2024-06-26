# Import packages
import keras.activations
import numpy as np
from keras.models import Sequential
from keras.layers import Masking, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from keras.utils import to_categorical


# Class
class Conv1dNoHybridNetwork:

    # Define class attributes
    mask_value = 0.0

    def __init__(self, num_classes=2, conv2_flag=False, binary_output=False):

        # Define attributes
        self.num_classes = num_classes
        if num_classes == 2:
            self.layer_dims = [16, 32]
            if not binary_output:
                self.output_neurons = 1
            else:
                self.output_neurons = 2
        else:
            self.layer_dims = [64, 64]
            self.output_neurons = num_classes

        # Layers
        self.conv2_flag = conv2_flag
        self.mask = Masking(mask_value=self.mask_value)
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
        self.model.add(GlobalAveragePooling1D(data_format="channels_last", keepdims=False))
        self.model.add(self.dense)

    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    def train(self, x, y, epochs, batch_size, show_epochs=False):
        if self.conv2_flag:
            x = np.expand_dims(x, 3)

        if self.output_neurons > 1:
            y = to_categorical(y, self.output_neurons)

        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2 * int(show_epochs))
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

        if self.output_neurons > 1:
            y = to_categorical(y, self.output_neurons)

        return self.model.evaluate(x, y, verbose=0)
