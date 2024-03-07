# Import packages
import numpy as np
from keras.models import Sequential
from keras.layers import Masking, Conv1D, MaxPooling1D, Flatten, Dense


# Class
class Conv1dNoHybridNetwork:

    # Define class attributes
    layer_dims = [16, 32]
    hidden_dim = 32
    mask_value = 0.0

    def __init__(self):
        # Layers
        self.conv2_flag = False
        self.mask = Masking(mask_value=self.mask_value)
        self.flat = Flatten()
        self.dense = Dense(1, activation="sigmoid")

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
        if self.conv2_flag:
            x = np.expand_dims(x, 3)

        return self.model.predict(x, verbose=0)

    def evaluate(self, x, y):
        if self.conv2_flag:
            x = np.expand_dims(x, 3)

        return self.model.evaluate(x, y, verbose=0)