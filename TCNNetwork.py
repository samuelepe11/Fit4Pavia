# Import packages
from keras.models import Sequential
from tcn import TCN

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class TCNNetwork(Conv1dNoHybridNetwork):

    def __init__(self, num_classes=2, binary_output=False):
        super().__init__(num_classes, binary_output=binary_output)

        # Layers
        self.model = Sequential()
        self.model.add(self.mask)
        self.model.add(TCN())
        self.model.add(self.dense)

    @staticmethod
    def show_structure(tcn_layer):
        for block in tcn_layer.__dict__["residual_blocks"]:
            print("    ", block.name)

