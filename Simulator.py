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
from RehabSkeletonDataset import RehabSkeletonDataset
from StatsHolder import StatsHolder
from Trainer import Trainer
from SimpleClassifierTrainer import SimpleClassifierTrainer
from FeatureExtractor import FeatureExtractor
from MLAlgorithmType import MLAlgorithmType


# Class
class Simulator:
    results_fold = Trainer.results_fold

    def __init__(self, desired_classes, n_rep, simulator_name, working_dir, folder_name, model_type, train_perc,
                 data_group_dict=None, train_epochs=None, train_lr=None, feature_file=None, normalize_data=False,
                 use_cuda=True, is_rehab=False):
        # Initialize attributes
        self.desired_classes = desired_classes
        self.n_rep = n_rep
        self.simulator_name = simulator_name
        self.working_dir = working_dir
        self.folder_name = folder_name
        if is_rehab:
            self.folder_name = "rehab_" + self.folder_name
            self.results_fold = "../IntelliRehabDS/" + self.results_fold
        self.results_dir = working_dir + self.results_fold + self.folder_name + "/"
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
        self.use_cuda = torch.cuda.is_available() and use_cuda

        self.train_stats = []
        self.test_stats = []
        self.mean_train_stats = None
        self.mean_test_stats = None
        self.dev_train_stats = None
        self.dev_test_stats = None
        self.train_cm_list = []
        self.train_cm_avg = None
        self.test_cm_list = []
        self.test_cm_avg = None

        # Create a folder to store models
        if self.folder_name not in os.listdir(working_dir + self.results_fold):
            os.mkdir(working_dir + self.results_fold + self.folder_name)
        if simulator_name not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + simulator_name)

        self.is_rehab = is_rehab

    def run_simulation(self, seed, keep_previous_results=False, is_rehab=False):
        # Define seeds
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cuda.deterministic = True
        keras.utils.set_random_seed(seed)
        silence_tensorflow()

        if keep_previous_results:
            previous_results = os.listdir(self.results_dir + self.simulator_name)
            self.reload_simulation_results()

        for i in range(self.n_rep):
            # Divide the dataset: if feature
            train_data, test_data = self.divide_dataset()

            # Train model
            if keep_previous_results:
                if "trial_" + str(i) + ".pt" in previous_results:
                    print("Trial", i, "already present...")
                    continue
            model_name = self.simulator_name + "/trial_" + str(i)

            if isinstance(self.model_type, NetType):
                trainer = NetworkTrainer(net_type=self.model_type, working_dir=self.working_dir,
                                         folder_name=self.folder_name, train_data=train_data, test_data=test_data,
                                         epochs=self.train_epochs, lr=self.train_lr,
                                         normalize_input=self.normalize_data, use_cuda=self.use_cuda,
                                         is_rehab=self.is_rehab)
            else:
                trainer = SimpleClassifierTrainer(ml_algorithm=self.model_type, working_dir=self.working_dir,
                                                  folder_name=self.folder_name, train_data=train_data,
                                                  test_data=test_data, normalize_data=self.normalize_data,
                                                  desired_classes=self.desired_classes, is_rehab=self.is_rehab)
            trainer.train(model_name, is_rehab=is_rehab)

            # Store results
            self.store_model_results(trainer)

        # Compute average confusion matrix
        self.compute_confusion_matrix()

    def compute_confusion_matrix(self):
        self.train_cm_avg = (np.round(np.mean(self.train_cm_list, axis=0))).astype(int)
        Trainer.draw_multiclass_confusion_matrix(self.train_cm_avg, self.desired_classes,
                                                 self.results_dir + self.simulator_name + "_train_cm.png",
                                                 is_rehab=self.is_rehab)
        self.test_cm_avg = (np.round(np.mean(self.test_cm_list, axis=0))).astype(int)
        Trainer.draw_multiclass_confusion_matrix(self.test_cm_avg, self.desired_classes,
                                                 self.results_dir + self.simulator_name + "_test_cm.png",
                                                 is_rehab=self.is_rehab)

    def store_model_results(self, trainer, avoid_eval=False):
        train_stats = trainer.test(set_type=SetType.TRAINING, show_cm=False, avoid_eval=avoid_eval)
        self.train_stats.append(train_stats)
        self.train_cm_list.append(trainer.train_cm)

        test_stats = trainer.test(set_type=SetType.TEST, show_cm=False, avoid_eval=avoid_eval)
        self.test_stats.append(test_stats)
        self.test_cm_list.append(trainer.test_cm)

    def reload_simulation_results(self, avoid_eval=False, is_rehab=False):
        for file in os.listdir(self.results_dir + self.simulator_name):
            if "net" in file:
                # Avoid opening the network information in the TCN folders
                continue

            trainer = Trainer.load_model(self.working_dir, self.folder_name, self.simulator_name + "/" +
                                         file.removesuffix(".pt"), self.use_keras, is_rehab=is_rehab)
            self.store_model_results(trainer, avoid_eval=avoid_eval)
            print("Information associated to " + file + " updated!")
        self.compute_confusion_matrix()

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
                                                 self.use_keras, is_rehab=self.is_rehab)
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
            if not self.is_rehab:
                train_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                             group_dict=self.data_group_dict, data_perc=self.train_perc)
                test_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                            data_names=train_data.remaining_instances)
            else:
                train_data = RehabSkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                                  data_perc=self.train_perc, maximum_length=self.data_group_dict)
                test_data = RehabSkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                                 data_names=train_data.remaining_instances)
        else:
            # Read feature data
            data_matrix, dim = FeatureExtractor.read_feature_file(working_dir=self.working_dir,
                                                                  feature_file=self.feature_file,
                                                                  group_dict=self.data_group_dict,
                                                                  is_rehab=self.is_rehab)

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
    def load_simulator(working_dir, folder_name, simulator_name, is_rehab=False):
        results_fold = Simulator.results_fold
        if is_rehab:
            folder_name = "rehab_" + folder_name
            results_fold = "../IntelliRehabDS/" + results_fold
        filepath = working_dir + results_fold + folder_name + "/" + simulator_name + ".pt"
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
            print("Pearson's correlation coefficient between " + name_x + " and " + name_y + ":", np.round(corr, 3))
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
    # desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]
    desired_classes1 = [1, 2]  # IntelliRehabDS correctness
    # desired_classes1 = list(range(3, 12))  # IntelliRehabDS gesture

    is_rehab1 = True
    # data_group_dict1 = {"C": 2, "R": 2}
    data_group_dict1 = 200
    model_type1 = NetType.TCN
    # model_type1 = MLAlgorithmType.AB
    train_perc1 = 0.7
    n_rep1 = 100
    train_epochs1 = 300
    # train_lr1 = 0.01  # Binary or Multiclass Conv2DNoHybrid
    # train_lr1 = 0.001  # Multiclass Conv2D or Conv1DNoHybrid or TCN or LSTMs
    # train_lr1 = 0.0001  # Multiclass Conv1D
    train_lr1 = None
    folder_name1 = "patientVSrandom_division_tcn"
    simulator_name1 = "random_division"
    use_cuda1 = False

    feature_file1 = "hand_crafted_features_global.csv"
    normalize_data1 = True

    # Initialize the simulator
    simulator1 = Simulator(desired_classes=desired_classes1, n_rep=n_rep1, simulator_name=simulator_name1,
                           working_dir=working_dir1, folder_name=folder_name1, data_group_dict=data_group_dict1,
                           model_type=model_type1, train_perc=train_perc1, train_epochs=train_epochs1,
                           train_lr=train_lr1, normalize_data=normalize_data1, use_cuda=use_cuda1, is_rehab=is_rehab1)
    '''simulator1 = Simulator(desired_classes=desired_classes1, n_rep=n_rep1, simulator_name=simulator_name1,
                           working_dir=working_dir1, folder_name=folder_name1, data_group_dict=data_group_dict1,
                           model_type=model_type1, train_perc=train_perc1, feature_file=feature_file1,
                           normalize_data=normalize_data1, is_rehab=is_rehab1)'''

    # Load simulator
    # simulator1 = Simulator.load_simulator(working_dir1, folder_name1, simulator_name1, is_rehab=is_rehab1)

    # Run simulation
    keep_previous_results1 = False
    simulator1.run_simulation(seed1, keep_previous_results=keep_previous_results1, is_rehab=is_rehab1)

    # Reload simulation results (in case of substantial modifications to the computed statistics)
    avoid_eval1 = False
    # simulator1.reload_simulation_results(avoid_eval=avoid_eval1, is_rehab=is_rehab1)

    # Assess and store simulator
    simulator1.assess_simulation(ci_alpha=0.05)
    Simulator.save_simulator(simulator1, simulator_name1)
    simulator1.log_simulation_results()
