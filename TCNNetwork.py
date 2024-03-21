# Import packages
from keras.models import Sequential
from tcn import TCN

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class TCNNetwork(Conv1dNoHybridNetwork):

    def __init__(self, output_neurons=1):
        super().__init__(output_neurons)

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        self.model.add(TCN())
        self.model.add(self.dense)
