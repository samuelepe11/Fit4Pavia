# Import packages
import numpy as np
from scipy.stats import pearsonr
from scipy.special import kl_div
from dtaidistance import dtw

from FeatureExtractor import FeatureExtractor
from SkeletonDataset import SkeletonDataset
from RehabSkeletonDataset import RehabSkeletonDataset


# Class
class VarianceAnalysisFeatureExtractor(FeatureExtractor):
    # Define class attributes
    data_processing_fold = "results/data_processing/"
    feature_names = ["features_normalised_distance", "signals_normalised_cross_correlation", "signals_kl_divergence",
                     "signals_dtw_distance"]
    feature_names = ["signals_kl_divergence", "signals_dtw_distance"]

    def __init__(self, working_dir, feature_input_file, dataset, group_dict=None, is_rehab=False):
        super().__init__(working_dir, None, dataset, None, False, None,
                         is_rehab)

        self.feature_input_file = feature_input_file
        self.features_dataset, self.dim = self.read_feature_file(working_dir, feature_input_file,
                                                                 only_descriptors=False, group_dict=group_dict,
                                                                 is_rehab=is_rehab)

    def get_file_headers(self):
        return ""

    def build_feature_datasets(self):
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

    @staticmethod
    def get_probability_distribution(x, bins=100):
        hist, bin_edges = np.histogram(x, bins=bins, density=True)
        return hist + 1e-10


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    # desired_classes1 = [8, 9]  # NTU HAR binary
    desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]  # NTU HAR multiclass
    # desired_classes1 = [1, 2]  # IntelliRehabDS correctness

    # feature_file1 = "hand_crafted_features_global_all.csv"
    feature_file1 = "hand_crafted_features_global_15classes.csv"
    is_rehab1 = False
    group_dict1 = {"C": 2, "R": 2} if not is_rehab1 else 200

    # Define dataset instance
    dataset1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, group_dict=group_dict1)
    # dataset1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
    #                                 maximum_length=group_dict1)

    # Build feature dataset
    feature_extractor1 = VarianceAnalysisFeatureExtractor(working_dir1, feature_file1, dataset1, group_dict1,
                                                          is_rehab=False)
    feature_extractor1.build_feature_datasets()

    # Load data
    data1, dim1 = FeatureExtractor.read_feature_file(working_dir=working_dir1, feature_file=feature_file1,
                                                     group_dict=group_dict1, is_rehab=is_rehab1)
