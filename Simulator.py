# Import packages
import os
import random
import torch
import pickle
import numpy as np
import scipy
import keras
from silence_tensorflow import silence_tensorflow

from NetworkTrainer import NetworkTrainer
from SetType import SetType
from NetType import NetType
from SkeletonDataset import SkeletonDataset
from StatsHolder import StatsHolder
from Trainer import Trainer
from SimpleClassifierTrainer import SimpleClassifierTrainer
from FeatureExtractor import FeatureExtractor
from MLAlgorithmType import MLAlgorithmType


# Class
class Simulator:
    results_fold = Trainer.results_fold

    def __init__(self, desired_classes, n_rep, simulator_name, working_dir, folder_name, model_type, train_perc,
                 data_group_dict=None, train_epochs=None, train_lr=None, feature_file=None, normalize_data=False):
        # Initialize attributes
        self.desired_classes = desired_classes
        self.n_rep = n_rep
        self.simulator_name = simulator_name
        self.working_dir = working_dir
        self.folder_name = folder_name
        self.results_dir = working_dir + self.results_fold + folder_name + "/"
        self.data_group_dict = data_group_dict

        self.model_type = model_type
        if model_type in NetworkTrainer.keras_networks:
            self.use_keras = True
        else:
            self.use_keras = False

        self.train_perc = train_perc
        self.train_epochs = train_epochs
        self.train_lr = train_lr
        self.feature_file = feature_file
        self.normalize_data = normalize_data

        self.train_stats = []
        self.test_stats = []
        self.mean_train_stats = None
        self.mean_test_stats = None
        self.dev_train_stats = None
        self.dev_test_stats = None

        # Create a folder to store models
        if folder_name not in os.listdir(working_dir + self.results_fold):
            os.mkdir(working_dir + self.results_fold + folder_name)
        if simulator_name not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + simulator_name)

    def run_simulation(self, seed):
        # Define seeds
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cuda.deterministic = True
        keras.utils.set_random_seed(seed)
        silence_tensorflow()

        for i in range(self.n_rep):
            # Divide the dataset: if feature
            train_data, test_data = self.divide_dataset()

            # Train model
            model_name = self.simulator_name + "/trial_" + str(i)
            if isinstance(self.model_type, NetType):
                trainer = NetworkTrainer(net_type=self.model_type, working_dir=self.working_dir,
                                         folder_name=self.folder_name, train_data=train_data, test_data=test_data,
                                         epochs=self.train_epochs, lr=self.train_lr,
                                         normalize_input=self.normalize_data)
            else:
                trainer = SimpleClassifierTrainer(ml_algorithm=self.model_type, working_dir=self.working_dir,
                                                  folder_name=self.folder_name, train_data=train_data,
                                                  test_data=test_data, normalize_data=self.normalize_data)
            trainer.train(model_name)

            # Store results
            self.store_model_results(trainer)

    def store_model_results(self, trainer):
        train_stats = trainer.test(set_type=SetType.TRAINING)
        self.train_stats.append(train_stats)
        test_stats = trainer.test(set_type=SetType.TEST)
        self.test_stats.append(test_stats)

    def reload_simulation_results(self):
        for file in os.listdir(self.results_dir + self.simulator_name):
            trainer = Trainer.load_model(self.working_dir, self.folder_name, self.simulator_name + "/" +
                                         file.removesuffix(".pt"), self.use_keras)
            self.store_model_results(trainer)
            print("Information associated to " + file + " updated!")

    def log_simulation_results(self, model_type=None):
        # Handle some test versions
        if self.results_dir[-2] == "1":
            self.results_dir = self.results_dir[:-2] + "/"
            self.folder_name = self.folder_name[:-1]
        prev_working_dir = "C:/Users/samue/OneDrive/Desktop/Files/Dottorato/Fit4Pavia/read_ntu_rgbd/"
        if self.results_dir.startswith(prev_working_dir):
            self.working_dir = "./../"
            self.results_dir = self.working_dir + self.results_dir.removeprefix(prev_working_dir)
        if model_type is not None:
            self.model_type = model_type

        n_trials = len(os.listdir(self.results_dir + self.simulator_name))
        try:
            if self.model_type == NetType.TCN:
                n_trials = int(n_trials / 2)
        except AttributeError:
            # Handle previous versions of the Simulator class (no model_type attribute)
            pass

        with open(self.results_dir + self.simulator_name + "_log.csv", "w") as f:
            f.write("run_id; dim_train; dim_test; train_acc; test_acc; train_f1; test_f1\n")
            train_dims = []
            test_accuracies = []
            test_f1s = []
            for ind in range(n_trials):
                file = "trial_" + str(ind)
                try:
                    trainer = Trainer.load_model(self.working_dir, self.folder_name, self.simulator_name + "/" + file,
                                                 self.use_keras)
                # Handle previous versions of the PatientDivisionSimulator class (no use_keras attribute)
                except AttributeError:
                    trainer = Trainer.load_model(self.working_dir, self.folder_name, self.simulator_name + "/" + file)

                train_data = trainer.train_data
                test_data = trainer.test_data

                try:
                    if self.use_keras:
                        train_data, _ = train_data
                        test_data, _ = test_data
                except AttributeError:
                    # The class has been defined and saved before the introduction of use_keras
                    self.use_keras = False

                if type(train_data) is tuple:
                    train_data, _ = train_data
                    test_data, _ = test_data

                dim_train = len(train_data)
                train_dims.append(dim_train)
                dim_test = len(test_data)
                train_acc = round(self.train_stats[ind].acc, 4)
                test_acc = round(self.test_stats[ind].acc, 4)
                test_accuracies.append(test_acc)
                train_f1 = round(self.train_stats[ind].f1, 4)
                test_f1 = round(self.test_stats[ind].f1, 4)
                test_f1s.append(test_f1)

                strings = [ind, dim_train, dim_test, train_acc, test_acc, train_f1, test_f1]
                f.write(";".join(map(str, strings)) + "\n")
            f.close()

            # Compute correlation between training set dimension and performance measures
            Simulator.assess_correlation(list_x=train_dims, list_y=test_accuracies, name_x="training set dimension",
                                         name_y="test accuracy", alpha=0.05)
            print()
            Simulator.assess_correlation(list_x=train_dims, list_y=test_f1s, name_x="training set dimension",
                                         name_y="test F1-score", alpha=0.05)

    def assess_simulation(self, ci_alpha):
        # Print average values
        self.mean_train_stats, self.dev_train_stats = StatsHolder.average_stats(self.train_stats)
        self.mean_test_stats, self.dev_test_stats = StatsHolder.average_stats(self.test_stats)

        # Print 95% confidence intervals
        print("CIs on the training set:")
        self.mean_train_stats.print_ci(ci_alpha)
        print()
        print("CIs on the test set:")
        self.mean_test_stats.print_ci(ci_alpha)

    def divide_dataset(self):
        # Perform a random division
        if self.feature_file is None or self.model_type == MLAlgorithmType.KNN_DTW:
            # Read skeleton data
            train_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                         group_dict=self.data_group_dict, data_perc=self.train_perc)
            test_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                        data_names=train_data.remaining_instances)
        else:
            # Read feature data
            data_matrix, dim = FeatureExtractor.read_feature_file(working_dir=self.working_dir,
                                                                  feature_file=self.feature_file,
                                                                  group_dict=self.data_group_dict)

            # Divide the dataset
            num_train_pt = round(dim * self.train_perc)
            ind = random.sample(range(dim), num_train_pt)
            train_data, test_data = Simulator.divide_features(data_matrix, dim, ind)

        return train_data, test_data

    @staticmethod
    def divide_features(data_matrix, dim, ind):
        train_data = data_matrix[ind, :]
        ind_test = [i for i in range(dim) if i not in ind]
        test_data = data_matrix[ind_test, :]

        return train_data, test_data

    @staticmethod
    def save_simulator(simulator, simulator_name):
        file_path = simulator.results_dir + simulator_name + ".pt"
        with open(file_path, "wb") as file:
            pickle.dump(simulator, file)
            print("'" + simulator_name + ".pt' has been successfully saved!")

    @staticmethod
    def load_simulator(working_dir, folder_name, simulator_name):
        filepath = working_dir + Simulator.results_fold + folder_name + "/" + simulator_name + ".pt"
        with open(filepath, "rb") as file:
            simulator = pickle.load(file)
        return simulator

    @staticmethod
    def assess_correlation(list_x, list_y, name_x, name_y, alpha):
        if len(np.unique(list_y)) == 1:
            # Avoid issues related to identical output stats
            corr = 0.0
        else:
            corr_mat = np.corrcoef(np.asarray(list_x), np.asarray(list_y))
            corr = corr_mat[0, 1]

        if not np.isnan(corr):
            print("Pearson's correlation coefficient between " + name_x + " and " + name_y + ":", corr)
            df = len(list_x) - 2
            t = corr / (np.sqrt(1 - corr ** 2 / df))

            if t >= 0:
                p_value = 2 * (1 - scipy.stats.t.cdf(t, df))
            else:
                p_value = 2 * scipy.stats.t.cdf(t, df)

            if p_value > alpha:
                addon = "NOT "
            else:
                addon = ""
            print("The " + name_x + " and the " + name_y + " ARE " + addon +
                  "correlated (p-value: {:.2e}".format(p_value) + ")")


# Main
if __name__ == "__main__":
    # Define variables
    seed1 = 111099
    working_dir1 = "./../"
    # desired_classes1 = [8, 9]
    desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 69, 70, 71, 80, 99]

    data_group_dict1 = {"C": 2, "R": 2}
    model_type1 = NetType.CONV1D
    # model_type1 = MLAlgorithmType.AB
    train_perc1 = 0.7
    n_rep1 = 100
    train_epochs1 = 300
    # train_lr1 = 0.01
    train_lr1 = 0.001
    folder_name1 = "patientVSrandom_division_conv1d_15classes"
    simulator_name1 = "sit_random_division"

    feature_file1 = "hand_crafted_features_global_10classes.csv"
    normalize_data1 = True

    # Initialize the simulator
    simulator1 = Simulator(desired_classes=desired_classes1, n_rep=n_rep1, simulator_name=simulator_name1,
                           working_dir=working_dir1, folder_name=folder_name1, data_group_dict=data_group_dict1,
                           model_type=model_type1, train_perc=train_perc1, train_epochs=train_epochs1,
                           train_lr=train_lr1, normalize_data=normalize_data1)
    # simulator1 = Simulator(desired_classes=desired_classes1, n_rep=n_rep1, simulator_name=simulator_name1,
    #                        working_dir=working_dir1, folder_name=folder_name1, data_group_dict=data_group_dict1,
    #                        model_type=model_type1, train_perc=train_perc1, feature_file=feature_file1,
    #                        normalize_data=normalize_data1)

    # Load simulator
    # simulator1 = Simulator.load_simulator(working_dir1, folder_name1, simulator_name1)

    # Run simulation
    simulator1.run_simulation(seed1)

    # Reload simulation results (in case of substantial modifications to the computed statistics)
    # simulator1.reload_simulation_results()

    # Assess and store simulator
    simulator1.assess_simulation(ci_alpha=0.05)
    Simulator.save_simulator(simulator1, simulator_name1)
    simulator1.log_simulation_results()
