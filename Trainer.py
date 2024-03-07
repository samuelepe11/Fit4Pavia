# Import packages
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import dill

from SetType import SetType
from StatsHolder import StatsHolder


# Class
class Trainer:
    results_fold = "results/models/"

    def __init__(self, working_dir, folder_name, train_data, test_data):
        # Initialize attributes
        self.train_losses = []

        # Define data parameters
        self.working_dir = working_dir
        self.folder_name = folder_name
        self.results_dir = working_dir + self.results_fold + folder_name + "/"
        self.train_data = train_data
        self.test_data = test_data

    def test(self, set_type=SetType.TRAINING):
        print("The model has not been defined.")
        train_stats = StatsHolder(float("inf"), 0, 0, 0, 0, 0)
        return train_stats

    def summarize_performance(self, show_process=False):
        # Show final losses
        train_stats = self.test(set_type=SetType.TRAINING)
        print("Train loss = " + str(round(train_stats.loss, 5)) + " - Train accuracy = " + str(round(train_stats.acc *
                                                                                                     100, 7)) + "%")
        test_stats = self.test(set_type=SetType.TEST)
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
    def compute_confusion_matrix(y_true, y_predicted):
        tp = (y_predicted == 1) & (y_true == 1)
        tn = (y_predicted == 0) & (y_true == 0)
        fp = (y_predicted == 1) & (y_true == 0)
        fn = (y_predicted == 0) & (y_true == 1)

        try:
            out = [np.sum(tp), np.sum(tn), np.sum(fp), np.sum(fn)]
        except:
            out = [torch.sum(tp).item(), torch.sum(tn).item(), torch.sum(fp).item(), torch.sum(fn).item()]

        return out

    @staticmethod
    def save_model(trainer, model_name, use_keras):
        file_path = trainer.results_dir + model_name + ".pt"
        with open(file_path, "wb") as file:
            if not use_keras:
                pickle.dump(trainer, file)
            else:
                dill.dump(trainer, file)

            print("'" + model_name + ".pt' has been successfully saved!... train loss: " +
                  str(np.round(trainer.train_losses[0], 4)) + " -> " + str(np.round(trainer.train_losses[-1], 4)))

    @staticmethod
    def load_model(working_dir, folder_name, model_name, use_keras=False):
        filepath = working_dir + Trainer.results_fold + folder_name + "/" + model_name + ".pt"
        with open(filepath, "rb") as file:
            if not use_keras:
                network_trainer = pickle.load(file)
            else:
                network_trainer = dill.load(file)
        return network_trainer
