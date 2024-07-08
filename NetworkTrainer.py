# Import packages
import torch
import torch.nn as nn
import random
import numpy as np
import keras
import time
from silence_tensorflow import silence_tensorflow
from tcn import TCN

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
    default_batch_size = 32
    keras_networks = [NetType.CONV1D_NO_HYBRID, NetType.CONV2D_NO_HYBRID, NetType.TCN]

    def __init__(self, net_type, working_dir, folder_name, train_data, test_data, epochs, lr,
                 batch_size=default_batch_size, binary_output=False, normalize_input=False, use_cuda=True):
        super().__init__(working_dir, folder_name, train_data, test_data)

        # Initialize attributes
        self.train_dim = len(self.train_data)
        self.test_dim = len(self.test_data)
        self.net_type = net_type
        self.start_time = None
        self.end_time = None

        self.num_classes = len(self.train_data.classes)
        self.binary_output = binary_output
        if self.num_classes > 2 or binary_output:
            self.multiclass = True
        else:
            self.multiclass = False

        self.normalize_input = normalize_input
        self.attr_mean = 0
        self.attr_std = 1

        self.descr_train = None
        self.descr_test = None
        if net_type == NetType.LSTM:
            self.net = LSTMNetwork(bidirectional=False, num_classes=self.num_classes)
        elif net_type == NetType.BLSTM:
            self.net = LSTMNetwork(bidirectional=True, num_classes=self.num_classes)
        elif net_type == NetType.GRU:
            self.net = GRUNetwork(num_classes=self.num_classes)
        elif net_type == NetType.CONV1D:
            self.net = Conv1dNetwork(num_classes=self.num_classes, binary_output=binary_output)
        elif net_type == NetType.CONV2D:
            self.net = Conv2dNetwork(num_classes=self.num_classes, binary_output=binary_output)
        elif net_type == NetType.RNN:
            self.net = RNNetwork(num_classes=self.num_classes)
        elif net_type == NetType.TRANS:
            self.net = TransformerNetwork(num_classes=self.num_classes)
        else:
            # Keras-based networks
            if net_type == NetType.CONV1D_NO_HYBRID:
                self.net = Conv1dNoHybridNetwork(num_classes=self.num_classes, binary_output=binary_output)
            elif net_type == NetType.TCN:
                self.net = TCNNetwork(num_classes=self.num_classes, binary_output=binary_output)
            else:
                self.net = Conv2dNoHybridNetwork(num_classes=self.num_classes, binary_output=binary_output)
            # Redefine datasets
            self.classes = self.train_data.classes
            self.descr_train = self.train_data.data_files
            self.descr_test = self.test_data.data_files
            self.train_data, self.test_data = SkeletonDataset.get_padded_dataset(self.train_data, self.test_data,
                                                                                 self.train_dim)

        # Define training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = None

        if self.net_type not in self.keras_networks:
            if not self.multiclass:
                self.criterion = nn.BCELoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr)
        else:
            if not self.multiclass:
                self.criterion = keras.losses.BinaryCrossentropy()
            else:
                self.criterion = keras.losses.CategoricalCrossentropy()
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
            self.net.compile(optimizer=self.optimizer, loss=self.criterion)

        self.use_cuda = torch.cuda.is_available() and use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def train(self, filename=None, show_epochs=False):
        self.model_name = filename
        if show_epochs:
            self.start_time = time.time()

        if self.use_cuda:
            net = NetworkTrainer.set_cuda(self.net)
            self.criterion = self.criterion.cuda()
        else:
            net = self.net

        if self.net_type not in self.keras_networks:
            use_keras = False

            if self.normalize_input:
                temp_train_data, self.attr_mean, self.attr_std = SkeletonDataset.normalize_data(self.train_data)
            else:
                temp_train_data = list(self.train_data)

            for epoch in range(self.epochs):
                self.set_training(True)
                train_loss = 0
                random.shuffle(temp_train_data)

                for x, y in temp_train_data:
                    x = torch.from_numpy(x)
                    x = x.unsqueeze(0)
                    x = x.to(self.device)

                    y = torch.tensor(y)
                    y = y.to(self.device)
                    if self.multiclass:
                        # Solve issues related to the CrossEntropyLoss
                        y = y.long()

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

                if show_epochs:
                    print("Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " completed... train loss = " +
                          str(np.round(train_loss, 5)))
        else:
            use_keras = True
            x, y = self.train_data
            history = net.train(x, y, epochs=self.epochs, batch_size=self.batch_size, show_epochs=show_epochs)
            self.train_losses = history.history["loss"]

        self.net = net
        Trainer.save_model(self, filename, use_keras=use_keras)

        if show_epochs:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print("Execution time:", round(duration / 60, 4), "min")

    def test(self, set_type=SetType.TRAINING, show_cm=False, avoid_eval=False, assess_calibration=False):
        if self.use_cuda:
            net = NetworkTrainer.set_cuda(self.net)
            self.criterion = self.criterion.cuda()
        else:
            net = self.net

        if set_type == SetType.TRAINING:
            data = self.train_data
            dim = self.train_dim
        else:
            data = self.test_data
            dim = self.test_dim

        # Store class labels
        if self.net_type not in self.keras_networks:
            class_labels = data.classes
            descr = data.data_files
            is_keras = False
        else:
            class_labels = self.classes
            descr = self.__dict__["descr_" + set_type.value]
            is_keras = True

        if self.normalize_input:
            data = SkeletonDataset.normalize_data(data, self.attr_mean, self.attr_std, is_keras=is_keras)

        y_true = []
        y_pred = []
        y_prob = []
        if self.net_type not in self.keras_networks:
            if not avoid_eval:  # Avoid issues with the binary models and the Conv1D and Conv2D multiclass models
                self.set_training(False)
            loss = 0
            with torch.no_grad():
                for x, y in data:
                    x = torch.from_numpy(x)
                    x = x.unsqueeze(0)
                    x = x.to(self.device)

                    y = torch.tensor(y)
                    y = y.to(self.device)

                    if self.multiclass:
                        # Solve issues related to the CrossEntropyLoss
                        y = y.long()

                    output = net(x, avoid_eval=avoid_eval)
                    output = output.squeeze()

                    # Cost function evaluation
                    temp_loss = self.criterion(output, y)
                    loss += temp_loss.item()

                    # Accuracy evaluation
                    if not self.multiclass:
                        prediction = (output >= 0.5).float()
                    else:
                        prediction = np.argmax(output)

                    # Store values for Confusion Matrix calculation
                    y_true.append(y)
                    y_pred.append(prediction)
                    y_prob.append(output)

            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            y_prob = np.asarray(y_prob)

            loss /= dim
            acc = np.sum(y_true == y_pred) / dim

        else:
            x, y = data
            prediction_prob = net.predict(x)
            if prediction_prob.shape[1] == 1:
                prediction_prob = prediction_prob.squeeze(1)
                prediction = np.round(prediction_prob)
            else:
                prediction = np.argmax(prediction_prob, axis=1)

            loss, acc = net.evaluate(x, y)

            # Store values for Confusion Matrix calculation
            y_true = y
            y_pred = prediction
            y_prob = prediction_prob

        cm = Trainer.compute_binary_confusion_matrix(y_true, y_pred, range(len(class_labels)))
        tp = cm[0]
        tn = cm[1]
        fp = cm[2]
        fn = cm[3]
        stats_holder = StatsHolder(loss, acc, tp, tn, fp, fn)

        if assess_calibration:
            stats_holder.calibration_results = self.assess_calibration(y_true, y_prob, y_pred, set_type, descr)

        # Compute multiclass confusion matrix
        cm_name = set_type.value + "_cm"
        if show_cm:
            img_path = self.results_dir + self.model_name + "/" + cm_name + ".png"
        else:
            img_path = None
        self.__dict__[cm_name] = Trainer.compute_multiclass_confusion_matrix(y_true, y_pred, class_labels, img_path)

        return stats_holder

    def show_model(self):
        print("DL MODEL:")

        if self.net_type not in self.keras_networks:
            attributes = self.net.__dict__
            for attr in attributes.keys():
                val = attributes[attr]
                if issubclass(type(val), nn.Module):
                    print(attr, "-" * (20 - len(attr)), val)
        else:
            layers = self.net.model.layers
            for layer in layers:
                print(" >", layer.name)

                if isinstance(layer, TCN):
                    TCNNetwork.show_structure(layer)

    def set_training(self, training=True):
        if training:
            self.net.train()
        else:
            self.net.eval()

        # Set specific layers
        for layer in self.net.__dict__.keys():
            if isinstance(self.net.__dict__[layer], nn.Module):
                # Set training/eval mode per each interested layer
                if "drop" in layer or "batch_norm" in layer:
                    self.net.__dict__[layer].training = training

    @staticmethod
    def set_cuda(net):
        net.cuda()
        # Set specific layers
        for layer in net.__dict__.keys():
            if isinstance(net.__dict__[layer], nn.Module):
                # Set cuda devise for parallelization
                net.__dict__[layer].cuda()

        return net


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
    # desired_classes1 = [8, 9]
    desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]

    # Define the data
    train_perc = 0.7
    train_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                  group_dict={"C": 2, "R": 2}, data_perc=train_perc)
    test_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                 data_names=train_data1.remaining_instances)

    # Define the model
    folder_name1 = "test"
    model_name1 = "conv1d"
    net_type1 = NetType.CONV1D
    binary_output1 = True
    normalize_input1 = False
    lr1 = 0.01  # Binary or Multiclass Conv2DNoHybrid
    # lr1 = 0.001  # Multiclass Conv2D or Conv1DNoHybrid or TCN
    # lr1 = 0.0001  # Multiclass Conv1D
    use_cuda1 = False
    show_cm1 = True
    assess_calibration1 = True
    trainer1 = NetworkTrainer(net_type=net_type1, working_dir=working_dir1, folder_name=folder_name1,
                              train_data=train_data1, test_data=test_data1, epochs=100, lr=lr1,
                              binary_output=binary_output1, normalize_input=normalize_input1, use_cuda=use_cuda1)

    # Train the model
    trainer1.summarize_performance()
    trainer1.train(model_name1, show_epochs=True)
    trainer1.summarize_performance(show_process=True, show_cm=show_cm1, assess_calibration=assess_calibration1)

    # Load trained model
    use_keras1 = True
    trainer1 = Trainer.load_model(working_dir=working_dir1, folder_name=folder_name1, model_name=model_name1,
                                  use_keras=use_keras1)

    avoid_eval1 = False
    trainer1.summarize_performance(show_process=True, show_cm=show_cm1, assess_calibration=assess_calibration1,
                                   avoid_eval=avoid_eval1)


