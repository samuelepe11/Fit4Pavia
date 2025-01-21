# Import packages
import os
import numpy as np
from scipy.special import kl_div
from dtaidistance import dtw

from FeatureExtractor import FeatureExtractor
from SkeletonDataset import SkeletonDataset
from RehabSkeletonDataset import RehabSkeletonDataset
from Trainer import Trainer
from Simulator import Simulator
from PatientDivisionSimulator import PatientDivisionSimulator
from SetType import SetType


# Class
class VarianceAnalysisFeatureExtractor(FeatureExtractor):
    # Define class attributes
    data_processing_fold = "results/data_processing/"
    models_folder = Trainer.results_fold
    feature_names = ["features_normalised_distance", "signals_normalised_cross_correlation", "signals_kl_divergence",
                     "signals_dtw_distance"]

    def __init__(self, working_dir, feature_input_file, dataset, group_dict=None, is_rehab=False):
        super().__init__(working_dir, None, dataset, None, False, None,
                         is_rehab)

        self.feature_input_file = feature_input_file
        self.features_dataset = None
        self.dim = None

        self.models_dir = working_dir + self.models_folder
        self.group_dict = group_dict
        self.n_classes = len(self.dataset.classes)

        self.distances = None
        self.descriptors = None

    def get_file_headers(self):
        return ""

    def build_feature_datasets(self):
        self.features_dataset, self.dim = self.read_feature_file(self.working_dir, self.feature_input_file,
                                                                 only_descriptors=False, group_dict=self.group_dict,
                                                                 is_rehab=self.is_rehab)
        for selected_feature in self.feature_names:
            self.build_feature_dataset(selected_feature, True)

    def build_feature_dataset(self, selected_feature=None, is_variance_analysis=False):
        is_signal = "signal" in selected_feature
        n = self.dim
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if is_signal:
                    x1, _ = self.dataset.__getitem__(i)
                    x2, _ = self.dataset.__getitem__(j)
                    x1, _, _ = self.normalize_data(x1)
                    x2, _, _ = self.normalize_data(x2)
                else:
                    x1 = self.features_dataset[i, :-1]
                    x2 = self.features_dataset[j, :-1]
                distance_matrix[i, j] = self.extract_features((x1, x2), selected_feature)
                distance_matrix[j, i] = distance_matrix[i, j]
                print(i, j)

        # Store the features into a .csv file
        self.raw_data = distance_matrix
        self.store_dataset(distance_matrix, self.descr, selected_feature + "_" + str(len(self.dataset.classes))
                           + "classes.csv", is_variance_analysis=is_variance_analysis)
        print("Raw dataset stored:", distance_matrix.shape)

    def extract_features(self, x, selected_feature=None):
        x1, x2 = x
        if selected_feature == "features_normalised_distance":
            feature_list = np.abs((x1 - x2) / (x1 + x2 + 1e-10))
        elif selected_feature == "signals_dtw_distance":
            feature_list = [dtw.distance(x1[:, i], x2[:, i]) for i in range(x1.shape[1])]
        elif selected_feature == "signals_kl_divergence":
            feature_list = []
            for i in range(x1.shape[1]):  # Iterate over channels
                p = self.get_probability_distribution(x1[:, i])
                q = self.get_probability_distribution(x2[:, i])
                feature_list.append(kl_div(p, q))
        elif selected_feature == "signals_normalised_cross_correlation":
            feature_list = [np.correlate(x1[:, i], x2[:, i], mode="full") / (np.std(x1[:, i]) * np.std(x2[:, i]) *
                                                                             len(x1[:, i])) for i in range(x1.shape[1])]
        else:
            print("Distance feature not available...")
            feature_list = [0]

        feature = np.mean(feature_list)
        return feature

    def aggregate_item_features(self, folder_name, use_keras=False, avoid_eval=False):
        # Read distance files
        if self.distances is None:
            self.distances = {selected_feature: FeatureExtractor.read_feature_file(working_dir=working_dir1,
                                                                                   feature_file=selected_feature,
                                                                                   group_dict=None,
                                                                                   is_rehab=is_rehab1,
                                                                                   variance_analysis_class_n=self.n_classes)[0]
                              for selected_feature in self.feature_names}
            self.descriptors = {selected_feature: list(FeatureExtractor.read_feature_file(working_dir=working_dir1,
                                                                                          feature_file=selected_feature,
                                                                                          group_dict=None,
                                                                                          only_descriptors=True,
                                                                                          is_rehab=is_rehab1,
                                                                                          variance_analysis_class_n=
                                                                                          self.n_classes)[0])
                                for selected_feature in self.feature_names}

        model_path = self.models_dir + folder_name
        with open(model_path + "/variance_analysis_features.csv", "w") as f:
            print("Processing features for " + folder_name + "...")
            # Write header
            f.write("item_name; simul_id; subj_id; dim_train; dim_test; is_cs; train_acc; test_acc; train_f1; test_f1; "
                    "avg_feature_dist; avg_cross_corr; avg_kl_dist; avg_dtw_dist; is_train_subj; is_test_subj; "
                    "pred_class; true_class; pred_confidence; is_pred_correct; item_loss\n")

            simulator_names = ["random_division", "patient_division"]
            if self.n_classes == 2:
                simulator_names = ["sit_" + s for s in simulator_names]

            for simulator_name in simulator_names:
                print(" " + simulator_name.replace("_", " ",).upper())
                # Load simulator
                try:
                    simulator = Simulator.load_simulator(self.working_dir, folder_name, simulator_name,
                                                         is_rehab=is_rehab1)
                # Handle previous versions of the Simulator class (no is_rehab attribute)
                except AttributeError:
                    simulator = Simulator.load_simulator(self.working_dir, folder_name, simulator_name)
                is_cs = "patient" in simulator_name

                fold = model_path + "/" + simulator_name
                n_trials = len(os.listdir(fold))
                for simul_id in range(n_trials):
                    train_acc = simulator.train_stats[simul_id].acc
                    test_acc = simulator.test_stats[simul_id].acc
                    train_f1 = simulator.train_stats[simul_id].f1
                    test_f1 = simulator.test_stats[simul_id].f1

                    # Load trainer
                    file = "trial_" + str(simul_id)
                    try:
                        trainer = Trainer.load_model(self.working_dir, folder_name, simulator_name + "/" + file,
                                                     use_keras, is_rehab=self.is_rehab,
                                                     feature_file=self.feature_input_file)
                    # Handle previous versions of the PatientDivisionSimulator class (no use_keras or is_rehab attribute)
                    except AttributeError:
                        trainer = Trainer.load_model(self.working_dir, folder_name, simulator_name + "/" + file)

                    try:
                        train_files = [file.strip(SkeletonDataset.extension) for file in trainer.train_data.data_files]
                        test_files = [file.strip(SkeletonDataset.extension) for file in trainer.test_data.data_files]
                    except:
                        train_files = [file.strip(SkeletonDataset.extension) for file in trainer.descr_train]
                        test_files = [file.strip(SkeletonDataset.extension) for file in trainer.descr_test]

                    # Get predictions
                    train_y_true, train_y_pred, train_y_prob = trainer.test(set_type=SetType.TRAINING,
                                                                            avoid_eval=avoid_eval, return_preds=True)
                    test_y_true, test_y_pred, test_y_prob = trainer.test(set_type=SetType.TEST, avoid_eval=avoid_eval,
                                                                         return_preds=True)
                    y_true = np.concatenate([train_y_true, test_y_true], 0)
                    y_pred = np.concatenate([train_y_pred, test_y_pred], 0)
                    y_prob = np.concatenate([train_y_prob, test_y_prob], 0)

                    # Characterize each item
                    dim_train = len(train_files)
                    dim_test = len(test_files)
                    all_list = train_files + test_files
                    for ind in range(len(all_list)):
                        item_name = all_list[ind]
                        subj_id = int(item_name[9:12])
                        pred_class = int(y_pred[ind])
                        true_class = int(y_true[ind])
                        pred_confidence = y_prob[ind]
                        is_pred_correct = int(pred_class == true_class)
                        item_loss = 1 - is_pred_correct
                        item_distances = {selected_feature: self.get_item_distances(selected_feature, item_name)
                                          for selected_feature in self.feature_names}

                        feat_dist = []
                        cc_dist = []
                        kl_dist = []
                        dtw_dist = []
                        for train_item_name in train_files:
                            feat_dist.append(self.get_distance(item_distances, "features_normalised_distance",
                                                               train_item_name))
                            cc_dist.append(self.get_distance(item_distances,
                                                             "signals_normalised_cross_correlation",
                                                             train_item_name))
                            kl_dist.append(self.get_distance(item_distances, "signals_kl_divergence",
                                                             train_item_name))
                            dtw_dist.append(self.get_distance(item_distances, "signals_dtw_distance",
                                                              train_item_name))
                        avg_feature_dist = np.nanmean(feat_dist)
                        avg_cross_corr = np.nanmean(cc_dist)
                        avg_kl_dist = np.nanmean(kl_dist)
                        avg_dtw_dist = np.nanmean(dtw_dist)

                        is_train_subj = int(item_name[9:12]) in SkeletonDataset.get_patient_ids(train_files)
                        is_test_subj = int(item_name[9:12]) in SkeletonDataset.get_patient_ids(test_files)
                        strings = [item_name, simul_id, subj_id, dim_train, dim_test, is_cs, train_acc, test_acc,
                                   train_f1, test_f1, avg_feature_dist, avg_cross_corr, avg_kl_dist, avg_dtw_dist,
                                   is_train_subj, is_test_subj, pred_class, true_class, pred_confidence,
                                   is_pred_correct, item_loss]
                        f.write(";".join(map(str, strings)) + "\n")
                    print(" - " + file + " loaded")
            f.close()

    def get_item_distances(self, feature_name, item_name):
        i = self.descriptors[feature_name].index(item_name)
        return self.distances[feature_name][i, :]

    def get_distance(self, item_distances, feature_name, item_name):
        item_distance = item_distances[feature_name]
        j = self.descriptors[feature_name].index(item_name)
        return item_distance[j]

    @staticmethod
    def get_probability_distribution(x, bins=100):
        hist, bin_edges = np.histogram(x, bins=bins, density=True)
        return hist + 1e-10


# Main
if __name__ == "__main__":
    # Define variables
    # working_dir1 = "./../"
    working_dir1 = "D:/Fit4Pavia/read_ntu_rgbd/"
    # desired_classes1 = [8, 9]  # NTU HAR binary
    # desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]  # NTU HAR multiclass
    desired_classes1 = [1, 2]  # IntelliRehabDS correctness

    feature_file1 = "hand_crafted_features_global_all.csv"
    feature_file1 = "hand_crafted_features_global.csv"
    # feature_file1 = "hand_crafted_features_global_15classes.csv"
    is_rehab1 = False
    group_dict1 = {"C": 2, "R": 2} if not is_rehab1 else 200

    # Define dataset instance
    dataset1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, group_dict=group_dict1)
    # dataset1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
    #                                 maximum_length=group_dict1)

    # Build signal-level feature dataset
    feature_extractor1 = VarianceAnalysisFeatureExtractor(working_dir1, feature_file1, dataset1, group_dict1,
                                                          is_rehab=False)
    # feature_extractor1.build_feature_datasets()

    # Load data
    # data1, dim1 = VarianceAnalysisFeatureExtractor.read_feature_file(working_dir=working_dir1,
    #                                                                  feature_file=feature_extractor1.feature_names[0],
    #                                                                  group_dict=group_dict1, is_rehab=is_rehab1,
    #                                                                  variance_analysis_class_n=len(desired_classes1))

    # Build item-level features
    folder_name1 = "patientVSrandom_division_svm"
    use_keras1 = False
    avoid_eval1 = False
    feature_extractor1.aggregate_item_features(folder_name1, use_keras=use_keras1, avoid_eval=avoid_eval1)
