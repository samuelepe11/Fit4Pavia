# Import packages
import torch
import torch.nn as nn
import random
import numpy as np
import keras
from silence_tensorflow import silence_tensorflow

from LSTMNetwork import LSTMNetwork
from GRUNetwork import GRUNetwork
from RNNetwork import RNNetwork
from Conv1dNetwork import Conv1dNetwork
from Conv2dNetwork import Conv2dNetwork
from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork
from Conv2dNoHybridNetwork import Conv2dNoHybridNetwork
from TCNNetwork import TCNNetwork
from TransformerNetwork import TransformerNetwork
from SetType import SetType
from NetType import NetType
from SkeletonDataset import SkeletonDataset
from StatsHolder import StatsHolder
from Trainer import Trainer


# Class
class NetworkTrainer(Trainer):
    keras_networks = [NetType.CONV1D_NO_HYBRID, NetType.CONV2D_NO_HYBRID, NetType.TCN]

    def __init__(self, net_type, working_dir, folder_name, train_data, test_data, epochs, lr, batch_size=32):
        super().__init__(working_dir, folder_name, train_data, test_data)

        # Initialize attributes
        self.train_dim = len(self.train_data)
        self.test_dim = len(self.test_data)
        self.net_type = net_type

        if net_type == NetType.LSTM:
            self.net = LSTMNetwork(bidirectional=False)
        elif net_type == NetType.BLSTM:
            self.net = LSTMNetwork(bidirectional=True)
        elif net_type == NetType.GRU:
            self.net = GRUNetwork()
        elif net_type == NetType.CONV1D:
            self.net = Conv1dNetwork()
        elif net_type == NetType.CONV2D:
            self.net = Conv2dNetwork()
        elif net_type == NetType.RNN:
            self.net = RNNetwork()
        elif net_type == NetType.TRANS:
            self.net = TransformerNetwork()
        else:
            # Keras-based networks
            if net_type == NetType.CONV1D_NO_HYBRID:
                self.net = Conv1dNoHybridNetwork()
            elif net_type == NetType.TCN:
                self.net = TCNNetwork()
            else:
                self.net = Conv2dNoHybridNetwork()
            # Redefine datasets
            self.train_data, self.test_data = SkeletonDataset.get_padded_datasets(self.train_data, self.test_data,
                                                                                  self.train_dim)

        # Define training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if self.net_type not in self.keras_networks:
            self.criterion = nn.BCELoss()
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr)
        else:
            self.criterion = keras.losses.BinaryCrossentropy()
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
            self.net.compile(optimizer=self.optimizer, loss=self.criterion)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def train(self, filename=None):
        if self.use_cuda:
            net = self.net.cuda()
            self.criterion = self.criterion.cuda()
        else:
            net = self.net

        if self.net_type not in self.keras_networks:
            use_keras = False
            net.train()
            temp_train_data = list(self.train_data)
            for epoch in range(self.epochs):
                train_loss = 0
                random.shuffle(temp_train_data)

                for x, y in temp_train_data:
                    x = torch.from_numpy(x)
                    x = x.unsqueeze(0)
                    x = x.to(self.device)

                    y = torch.tensor(y)
                    y = y.to(self.device)

                    output = net(x)
                    output = output.squeeze()

                    # Train loss evaluation
                    loss = self.criterion(output, y)
                    train_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss = train_loss / len(self.train_data)
                self.train_losses.append(train_loss)
        else:
            use_keras = True
            x, y = self.train_data
            history = net.train(x, y, epochs=self.epochs, batch_size=self.batch_size)
            self.train_losses = history.history["loss"]

        self.net = net
        Trainer.save_model(self, filename, use_keras=use_keras)

    def test(self, set_type=SetType.TRAINING):
        if self.use_cuda:
            net = self.net.cuda()
            self.criterion = self.criterion.cuda()
        else:
            net = self.net

        if set_type == SetType.TRAINING:
            data = self.train_data
            dim = self.train_dim
        else:
            data = self.test_data
            dim = self.test_dim

        if self.net_type not in self.keras_networks:
            net.eval()
            loss = 0
            acc = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            with torch.no_grad():
                for x, y in data:
                    x = torch.from_numpy(x)
                    x = x.unsqueeze(0)
                    x = x.to(self.device)

                    y = torch.tensor(y)
                    y = y.to(self.device)

                    output = net(x)
                    output = output.squeeze()

                    # Cost function evaluation
                    temp_loss = self.criterion(output, y)
                    loss += temp_loss.item()

                    # Accuracy evaluation
                    prediction = (output >= 0.5).float()
                    acc += (prediction == y).item()

                    # Confusion matrix definition
                    cm = Trainer.compute_confusion_matrix(y, prediction)
                    tp += cm[0]
                    tn += cm[1]
                    fp += cm[2]
                    fn += cm[3]

            loss /= dim
            acc /= dim
        else:
            x, y = data
            prediction = net.predict(x)
            prediction = prediction.squeeze(1)
            prediction = np.round(prediction)

            loss, acc = net.evaluate(x, y)
            cm = Trainer.compute_confusion_matrix(y, prediction)
            tp = cm[0]
            tn = cm[1]
            fp = cm[2]
            fn = cm[3]

        stats_holder = StatsHolder(loss, acc, tp, tn, fp, fn)
        return stats_holder


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 111099
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.deterministic = True
    keras.utils.set_random_seed(seed)
    silence_tensorflow()

    # Define variables
    working_dir1 = "./../"
    desired_classes1 = [8, 9]

    # Define the data
    train_perc = 0.7
    train_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                  group_dict={"C": 2, "R": 2}, data_perc=train_perc)
    test_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                 data_names=train_data1.remaining_instances)

    # Define the model
    folder_name1 = "tests"
    model_name1 = "test_refactor_conv2"
    net_type1 = NetType.CONV2D_NO_HYBRID
    trainer1 = NetworkTrainer(net_type=net_type1, working_dir=working_dir1, folder_name=folder_name1,
                              train_data=train_data1, test_data=test_data1, epochs=300, lr=0.01)

    # Train the model
    #trainer1.summarize_performance()
    trainer1.train(model_name1)
    trainer1.summarize_performance(show_process=True)

    # Load trained model
    # use_keras1 = True
    # trainer1 = NetworkTrainer.load_model(working_dir=working_dir1, folder_name=folder_name1, model_name=model_name1,
    #                                      use_keras=use_keras1)
    # trainer1.summarize_performance(show_process=True)
