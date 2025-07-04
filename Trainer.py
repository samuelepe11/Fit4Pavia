# Import packages
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import tensorflow as tf
import dill
import keras
import os
from torcheval.metrics.functional import multiclass_confusion_matrix
from pandas import DataFrame
from calfram.calibrationframework import select_probability, reliabilityplot, calibrationdiagnosis, classwise_calibration

from SetType import SetType
from StatsHolder import StatsHolder
from NetType import NetType
from TCNNetwork import TCNNetwork
from SkeletonDataset import SkeletonDataset
from RehabSkeletonDataset import RehabSkeletonDataset


# Class
class Trainer:
    results_fold = "results/models/"

    def __init__(self, working_dir, folder_name, train_data, test_data, is_rehab=False):
        # Initialize attributes
        self.train_losses = []
        self.train_accs = []

        if is_rehab:
            self.results_fold = "../IntelliRehabDS/" + self.results_fold

        # Define data parameters
        self.working_dir = working_dir
        self.folder_name = folder_name
        self.results_dir = working_dir + self.results_fold
        if folder_name not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + folder_name)
        self.results_dir += folder_name + "/"
        self.train_data = train_data
        self.test_data = test_data
        self.model_name = None

    def test(self, set_type=SetType.TRAINING, show_cm=False, avoid_eval=False, assess_calibration=False,
             return_only_preds=False, return_also_preds=False, is_rehab=False, ext_test_data=None, ext_test_name=None,
             for_generation=False):
        print("The model has not been defined.")
        train_stats = StatsHolder(float("inf"), 0, 0, 0, 0, 0)
        return train_stats

    def get_bootstrapped_metrics(self, preds, class_labels, n_rep=100, boot_dim=70, alpha_ci=5):
        return None, None, None, None

    def summarize_performance(self, show_process=False, show_cm=False, assess_calibration=False, avoid_eval=False,
                              is_rehab=False, ext_test_data_list=None, ext_test_name_list=None, for_generation=False):
        get_comparable_stat = ext_test_data_list is not None
        if show_cm or assess_calibration or show_process:
            if self.model_name not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + self.model_name)

        # Show final losses
        train_stats = self.test(set_type=SetType.TRAINING, show_cm=show_cm, assess_calibration=assess_calibration,
                                avoid_eval=avoid_eval, is_rehab=is_rehab, return_also_preds=get_comparable_stat,
                                for_generation=for_generation)
        if get_comparable_stat:
            means = {}
            stds = {}
            cis = {}
            stats = {}
            train_stats, preds = train_stats
            class_labels = self.train_data.classes if self.net_type not in self.keras_networks else self.classes

            mean_stats, std_stats, ci_dict, stats_list_dict = self.get_bootstrapped_metrics(preds, class_labels)
            means.update({"train": mean_stats})
            stds.update({"train": std_stats})
            cis.update({"train": ci_dict})
            stats.update({"train": stats_list_dict})

        print("Train loss = " + str(round(train_stats.loss, 5)) + " - Train accuracy = "
              + str(round(train_stats.acc * 100, 7)) + "%" + " - Train F1-score = " + str(round(train_stats.f1 * 100,
                                                                                                7)) + "%")
        if assess_calibration:
            Trainer.show_calibration_table(train_stats, "training")

        test_stats = self.test(set_type=SetType.TEST, show_cm=show_cm, assess_calibration=assess_calibration,
                               avoid_eval=avoid_eval, is_rehab=is_rehab, return_also_preds=get_comparable_stat,
                               for_generation=for_generation)
        if get_comparable_stat:
            test_stats, preds = test_stats
            mean_stats, std_stats, ci_dict, stats_list_dict = self.get_bootstrapped_metrics(preds, class_labels)
            means.update({"test": mean_stats})
            stds.update({"test": std_stats})
            cis.update({"test": ci_dict})
            stats.update({"test": stats_list_dict})

        print("Test loss = " + str(round(test_stats.loss, 5)) + " - Test accuracy = "
              + str(round(test_stats.acc * 100, 7)) + "%" + " - Test F1-score = " + str(round(test_stats.f1 * 100, 7))
              + "%")
        if assess_calibration:
            Trainer.show_calibration_table(test_stats, "test")

        # Show training curves
        if show_process:
            self.draw_training_curves()
            plt.savefig(self.results_dir + self.model_name + "/training_curves.png", dpi=300, bbox_inches="tight")
            plt.close()

        if ext_test_data_list is not None:
            for i, ext_test_data in enumerate(ext_test_data_list):
                ext_test_name = ext_test_name_list[i]
                ext_test_name = ext_test_name + "_" if ext_test_name is not None else ""
                ext_test_stats = self.test(set_type=SetType.EXT_TEST, show_cm=show_cm, assess_calibration=assess_calibration,
                                           avoid_eval=avoid_eval, is_rehab=is_rehab, ext_test_data=ext_test_data,
                                           ext_test_name=ext_test_name, return_also_preds=get_comparable_stat,
                                           for_generation=for_generation)
                if get_comparable_stat:
                    ext_test_stats, preds = ext_test_stats

                print(ext_test_name[:-1].upper() + ": External test loss = " + str(round(ext_test_stats.loss, 5)) + " - "
                      + "External test accuracy = " + str(round(ext_test_stats.acc * 100, 7)) + "%" + " - External test "
                      + "F1-score = " + str(round(ext_test_stats.f1 * 100, 7)) + "%")
                if assess_calibration:
                    Trainer.show_calibration_table(ext_test_stats, ext_test_name + "external test")

                if get_comparable_stat:
                    mean_stats, std_stats, ci_dict, stats_list_dict = self.get_bootstrapped_metrics(preds, class_labels)
                    means.update({ext_test_name[:-1]: mean_stats})
                    stds.update({ext_test_name[:-1]: std_stats})
                    cis.update({ext_test_name[:-1]: ci_dict})
                    stats.update({ext_test_name[:-1]: stats_list_dict})

            return means, stds, cis, stats

    def assess_calibration(self, y_true, y_prob, y_pred, set_type, descr=None, ext_test_name=None):
        class_scores = select_probability(y_true, y_prob, y_pred)

        # Store results file
        data = np.concatenate((y_true[:, np.newaxis], y_pred[:, np.newaxis], y_prob), axis=1)
        titles = ["y_true", "y_pred"] + ["y_prob" + str(i) for i in range(y_prob.shape[1])]
        if descr is not None:
            descr = np.asarray([d.strip(SkeletonDataset.extension) for d in descr])
            data = np.concatenate((descr[:, np.newaxis], data), axis=1)
            titles = ["descr"] + titles
        df = DataFrame(data, columns=titles)
        addon = "" if ext_test_name is None else ext_test_name
        df.to_csv(self.results_dir + self.model_name + "/" + addon + set_type.value + "_classification_results.csv",
                  index=False)

        # Draw reliability plot
        reliabilityplot(class_scores, strategy=10, split=False)
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.savefig(self.results_dir + self.model_name + "/" + addon + set_type.value + "_calibration.png")
        plt.close()

        # Compute local metrics
        results, _ = calibrationdiagnosis(class_scores, strategy=10)

        # Compute global metrics
        results_cw = classwise_calibration(results)
        return results_cw

    def draw_training_curves(self):
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.suptitle("Training curves")

        # Losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, "b", label="Training set")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

        # Accuracies
        plt.subplot(2, 1, 2)
        try:
            plt.plot(self.train_accs, "b", label="Training set")
            plt.legend()
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
        except:
            plt.xlabel("Accuracy values have not been recorded during training!")

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
    def compute_multiclass_confusion_matrix(y_true, y_pred, classes, img_path=None, is_rehab=False,
                                            for_generation=False):
        # Compute confusion matrix
        y_true = torch.tensor(y_true, dtype=torch.int64)
        y_pred = torch.tensor(y_pred, dtype=torch.int64)
        cm = multiclass_confusion_matrix(y_pred, y_true.to(torch.int64), len(classes))

        # Draw heatmap
        if img_path is not None:
            Trainer.draw_multiclass_confusion_matrix(cm, classes, img_path, is_rehab=is_rehab,
                                                     for_generation=for_generation)

        return cm

    @staticmethod
    def draw_multiclass_confusion_matrix(cm, classes, img_path, is_rehab=False, for_generation=False):
        plt.figure(figsize=(8, 8))
        if not is_rehab:
            actions = SkeletonDataset.actions
            rotation = 45
        else:
            actions = RehabSkeletonDataset.action_labels
            rotation = 15

        labels = [" ".join(actions[c - 1].split(" ")[:3]) for c in classes]

        plt.imshow(cm, cmap="Reds")
        if for_generation:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
        fontsize = "xx-large" if not for_generation or cm.shape[0] == 2 else "large"
        addon = "" if not for_generation else "%"
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j].item()
                if for_generation:
                    val = np.round(val * 100, 2)
                plt.text(j, i, f"{val}" + addon, ha="center", va="center", color="black", fontsize=fontsize)
        plt.xticks(range(len(classes)), labels, rotation=rotation)
        plt.xlabel("Predicted class")
        plt.yticks(range(len(classes)), labels, rotation=rotation)
        y_lab = "True class" if not for_generation else "Conditioning class"
        plt.ylabel(y_lab)
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def save_model(trainer, model_name, use_keras,
                   absolute_path="C:/Users/samue/OneDrive/Desktop/Files/Dottorato/Fit4Pavia/read_ntu_rgbd/",
                   is_rehab=False):
        file_path = trainer.results_dir + model_name + ".pt"
        with open(file_path, "wb") as file:
            if not use_keras:
                pickle.dump(trainer, file)
            else:
                if trainer.net_type == NetType.TCN:
                    # Store the network separately because dill is unable to store TCN layers
                    file_path_net = file_path.strip(".pt") + "_net.pt"
                    if is_rehab:
                        absolute_path = "C:/Users/samue/OneDrive/Desktop/Files/Dottorato/Fit4Pavia/"
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
    def load_model(working_dir, folder_name, model_name, use_keras=False, folder_path=None, is_rehab=False,
                   feature_file=None):
        if folder_name is not None:
            results_fold = Trainer.results_fold
            if is_rehab:
                results_fold = "../IntelliRehabDS/" + results_fold
            folder_path = working_dir + results_fold + folder_name + "/"
        filepath = folder_path + model_name + ".pt"

        with open(filepath, "rb") as file:
            if not use_keras:
                network_trainer = pickle.load(file)
            else:
                network_trainer = dill.load(file)

                if network_trainer.net_type == NetType.TCN:
                    if "classes" not in network_trainer.__dict__.keys():
                        # Handle previous versions of the NetworkTrainer class (no classes attribute)
                        network_trainer.classes = [8, 9]
                        network_trainer.num_classes = 2
                        network_trainer.binary_output = False

                    # Overcome issues related to separation of model and trainer during saving
                    file_path_net = filepath.strip(".pt") + "_net.pt"
                    if filepath.startswith(".") and not file_path_net.startswith("."):
                        # Strip removes the starting "." from the string along with the extension
                        file_path_net = "." + file_path_net
                    network_trainer.net = TCNNetwork(network_trainer.num_classes, network_trainer.binary_output)
                    network_trainer.net.compile(keras.optimizers.Adam(learning_rate=0.01), keras.losses.BinaryCrossentropy())
                    x, y = network_trainer.train_data
                    tf.random.set_seed(1)
                    network_trainer.net.train(x, y, 1, 1)
                    network_trainer.net.model.load_weights(file_path_net)

        if not is_rehab:
            if "model_name" not in network_trainer.__dict__.keys():
                # Handle previous versions of the Trainer classes (no model_name attribute)
                network_trainer.model_name = model_name

            if network_trainer.results_dir != folder_path:
                network_trainer.results_dir = folder_path
                network_trainer.model_name = model_name

            if "classes" not in network_trainer.__dict__.keys():
                # Handle previous versions of the NetworkTrainer class (no classes attribute)
                network_trainer.classes = [8, 9]

            try:
                if "num_classes" not in network_trainer.net.__dict__.keys():
                    # Handle previous versions of the Conv1dNetwork class (no num_classes attribute)
                    network_trainer.net.num_classes = 2
            except AttributeError:
                print()

            if (not isinstance(network_trainer.train_data, SkeletonDataset) and
                    "descr_train" not in network_trainer.__dict__.keys()):
                network_trainer.descr_train = network_trainer.find_data_files(network_trainer.train_data,
                                                                              feature_file)
                network_trainer.descr_test = network_trainer.find_data_files(network_trainer.test_data,
                                                                             feature_file)

            try:
                if "is_2d" not in network_trainer.net.__dict__.keys():
                    # Handle previous versions of the Conv1dNetwork class (no is_2d attribute)
                    network_trainer.net.is_2d = False

                if "num_classes" not in network_trainer.net.__dict__.keys():
                    # Handle previous versions of the Conv1dNetwork class (no num_classes attribute)
                    network_trainer.net.num_classes = 2

                if "binary_output" not in network_trainer.__dict__.keys() and "net_type" in network_trainer.__dict__.keys():
                    # Handle previous versions of the NetworkTrainer class (no binary_output attribute)
                    network_trainer.binary_output = True
            except:
                pass

            if "multiclass" not in network_trainer.__dict__.keys():
                # Handle previous versions of the NetworkTrainer class (no multiclass attribute)
                network_trainer.multiclass = False

            if "normalize_input" not in network_trainer.__dict__.keys():
                # Handle previous versions of the NetworkTrainer class (no normalize_input attribute)
                network_trainer.normalize_input = False

            if "normalize_data" not in network_trainer.__dict__.keys():
                # Handle previous versions of the SimpleClassifierTrainer class (no normalize_data attribute)
                network_trainer.normalize_data = False

            if "15" in network_trainer.model_name and len(network_trainer.classes) != 15:
                network_trainer.classes = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]

            if "15" not in network_trainer.model_name and len(network_trainer.classes) != 2:
                network_trainer.classes = [8, 9]

        return network_trainer

    @staticmethod
    def show_calibration_table(stats, set_name):
        print("Calibration information for", set_name.upper() + " set:")
        for stat in stats.calibration_results.keys():
            print(" - " + stat + ": " + str(stats.calibration_results[stat]))
