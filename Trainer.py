# Import packages
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import dill
import keras
import os
from torcheval.metrics.functional import multiclass_confusion_matrix

from SetType import SetType
from StatsHolder import StatsHolder
from NetType import NetType
from TCNNetwork import TCNNetwork
from SkeletonDataset import SkeletonDataset


# Class
class Trainer:
    results_fold = "results/models/"

    def __init__(self, working_dir, folder_name, train_data, test_data):
        # Initialize attributes
        self.train_losses = []

        # Define data parameters
        self.working_dir = working_dir
        self.folder_name = folder_name
        self.results_dir = working_dir + self.results_fold
        if folder_name not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + folder_name)
        self.results_dir += folder_name + "/"
        self.train_data = train_data
        self.test_data = test_data

    def test(self, set_type=SetType.TRAINING, show_cm=False):
        print("The model has not been defined.")
        train_stats = StatsHolder(float("inf"), 0, 0, 0, 0, 0)
        return train_stats

    def summarize_performance(self, show_process=False, show_cm=False):
        # Show final losses
        train_stats = self.test(set_type=SetType.TRAINING, show_cm=show_cm)
        print("Train loss = " + str(round(train_stats.loss, 5)) + " - Train accuracy = " + str(round(train_stats.acc *
                                                                                                     100, 7)) + "%")
        test_stats = self.test(set_type=SetType.TEST, show_cm=show_cm)
        print("Test loss = " + str(round(test_stats.loss, 5)) + " - Test accuracy = " + str(round(test_stats.acc * 100,
                                                                                                  7)) + "%")

        # Show training curves
        if show_process:
            plt.plot(self.train_losses, "b", label="Training set")
            plt.legend()
            plt.title("Training curves")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.show()

    @staticmethod
    def compute_binary_confusion_matrix(y_true, y_predicted, classes=None):
        if classes is None:
            # Classical binary computation (class 0 as negative and class 1 as positive)
            tp = (y_predicted == 1) & (y_true == 1)
            tn = (y_predicted == 0) & (y_true == 0)
            fp = (y_predicted == 1) & (y_true == 0)
            fn = (y_predicted == 0) & (y_true == 1)

            try:
                out = [np.sum(tp), np.sum(tn), np.sum(fp), np.sum(fn)]
            except:
                out = [torch.sum(tp).item(), torch.sum(tn).item(), torch.sum(fp).item(), torch.sum(fn).item()]

            return out
        else:
            # One VS Rest computation for Macro-Averaged F1-score and other metrics
            out = []
            for c in classes:
                y_true_i = (y_true == c).astype(int)
                y_predicted_i = (y_predicted == c).astype(int)
                out_i = Trainer.compute_binary_confusion_matrix(y_true_i, y_predicted_i, classes=None)
                out.append(out_i)

            out = np.asarray(out)
            out = [out[:, i] for i in range(out.shape[1])]
            return out

    @staticmethod
    def compute_multiclass_confusion_matrix(y_true, y_pred, classes, img_path=None):
        # Compute confusion matrix
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        cm = multiclass_confusion_matrix(y_pred, y_true, len(classes), normalize="pred")

        # Draw heatmap
        if img_path is not None:
            Trainer.draw_multiclass_confusion_matrix(cm, classes, img_path)

        return cm

    @staticmethod
    def draw_multiclass_confusion_matrix(cm, classes, img_path):
        plt.figure(figsize=(10, 10))
        labels = [" ".join(SkeletonDataset.actions[c - 1].split(" ")[:2]) for c in classes]

        plt.imshow(cm, cmap="jet")
        plt.xticks(range(len(classes)), labels, rotation=45)
        plt.yticks(range(len(classes)), labels, rotation=45)
        plt.savefig(img_path, dpi=300)
        plt.close()

    @staticmethod
    def save_model(trainer, model_name, use_keras,
                   absolute_path="C:/Users/samue/OneDrive/Desktop/Files/Dottorato/Fit4Pavia/read_ntu_rgbd/"):
        file_path = trainer.results_dir + model_name + ".pt"
        with open(file_path, "wb") as file:
            if not use_keras:
                pickle.dump(trainer, file)
            else:
                if trainer.net_type == NetType.TCN:
                    # Store the network separately because dill is unable to store TCN layers
                    file_path_net = file_path.strip(".pt") + "_net.pt"
                    file_path_net = absolute_path + file_path_net.strip("./../")
                    trainer.net.model.save(file_path_net)
                    trainer.net = "Empty"

                dill.dump(trainer, file)

                if trainer.net_type == NetType.TCN:
                    # Restore the network
                    trainer.net = TCNNetwork(trainer.num_classes, trainer.binary_output)
                    trainer.net.compile(trainer.optimizer, trainer.criterion)
                    trainer.net.model.load_weights(file_path_net)

            print("'" + model_name + ".pt' has been successfully saved!... train loss: " +
                  str(np.round(trainer.train_losses[0], 4)) + " -> " + str(np.round(trainer.train_losses[-1], 4)))

    @staticmethod
    def load_model(working_dir, folder_name, model_name, use_keras=False, folder_path=None):
        if folder_name is not None:
            filepath = working_dir + Trainer.results_fold + folder_name + "/"
        else:
            filepath = folder_path
        filepath += model_name + ".pt"

        with open(filepath, "rb") as file:
            if not use_keras:
                network_trainer = pickle.load(file)
            else:
                network_trainer = dill.load(file)

                if network_trainer.net_type == NetType.TCN:
                    # Overcome issues related to separation of model and trainer during saving
                    file_path_net = filepath.strip(".pt") + "_net.pt"
                    if filepath.startswith(".") and not file_path_net.startswith("."):
                        # Strip removes the starting "." from the string along with the extension
                        file_path_net = "." + file_path_net
                    network_trainer.net = TCNNetwork(network_trainer.num_classes, network_trainer.binary_output)
                    network_trainer.net.compile(keras.optimizers.Adam(learning_rate=0.01), keras.losses.BinaryCrossentropy())
                    x, y = network_trainer.train_data
                    network_trainer.net.train(x, y, 1, 1)
                    network_trainer.net.model.load_weights(file_path_net)

        if "is_2d" not in network_trainer.net.__dict__.keys():
            # Handle previous versions of the Conv1dNetwork class (no is_2d attribute)
            network_trainer.net.is_2d = False

        if "num_classes" not in network_trainer.net.__dict__.keys():
            # Handle previous versions of the Conv1dNetwork class (no num_classes attribute)
            network_trainer.net.num_classes = 2

        if "multiclass" not in network_trainer.__dict__.keys():
            # Handle previous versions of the NetworkTrainer class (no multiclass attribute)
            network_trainer.multiclass = False

        if "normalize_input" not in network_trainer.__dict__.keys():
            # Handle previous versions of the NetworkTrainer class (no normalize_input attribute)
            network_trainer.normalize_input = False

        return network_trainer
