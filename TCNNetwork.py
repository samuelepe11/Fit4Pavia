# Import packages
from keras.models import Sequential
from tcn import TCN

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class TCNNetwork(Conv1dNoHybridNetwork):

    def __init__(self, num_classes=2):
        super().__init__(num_classes)

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        self.model.add(TCN())
        self.model.add(self.dense)
