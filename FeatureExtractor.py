# Import packages
import numpy as np
from scipy.stats import hmean, median_abs_deviation, iqr, skew, kurtosis

from SkeletonDataset import SkeletonDataset
from RehabSkeletonDataset import RehabSkeletonDataset


# Class
class FeatureExtractor:
    # Define class attributes
    n_joints = 25
    data_processing_fold = "results/data_processing/"
    suffixes = ["_x", "_y", "_z"]
    feature_names = ["mean", "absolute_harmonic_mean", "std", "max", "min", "range", "median", "median_absolute_dev",
                     "iq_range", "sum_of_area", "mean_energy", "skewness", "kurtosis", "pearson_corr_coeff",
                     "mean_velocity"]

    def __init__(self, working_dir, feature_file, dataset, n_windows, include_l2=True, selected_features=None,
                 is_rehab=False):
        if is_rehab:
            self.data_processing_fold = "../IntelliRehabDS/" + self.data_processing_fold
            feature_file = "rehab_" + feature_file

        self.working_dir = working_dir
        self.data_processing_dir = working_dir + self.data_processing_fold
        self.feature_file = feature_file
        self.raw_data = None
        self.preprocessed_data = None

        self.dataset = dataset
        self.n_windows = n_windows

        if selected_features is not None:
            self.feature_names = selected_features

        if include_l2:
            self.suffixes.append("_l2")
        self.include_l2 = include_l2

        # Build file headers
        self.file_headers = self.get_file_headers()

        # Store data descriptors
        descr = list(self.dataset.data_files)
        descr = [x.strip(self.dataset.extension) for x in descr]
        self.descr = np.asarray(descr)
        self.preprocessed_descr = None

    def get_file_headers(self):
        header = []
        for joint in range(self.n_joints):
            for window in range(self.n_windows):
                for feat in self.feature_names:
                    for suffix in self.suffixes:
                        if suffix == "_l2" and feat == "pearson_corr_coeff":
                            continue
                        s = feat + suffix + "_joint" + str(joint + 1)
                        if self.n_windows != 1:
                            s += "_window" + str(window)
                        header.append(s)
        header.append("class")

        header = ";".join(header)
        return header

    def build_feature_dataset(self):
        data = []
        labels = []
        descr = []
        for i in range(self.dataset.len):
            x, y = self.dataset.__getitem__(i)
            features = self.extract_features(x)
            data.append(features)
            labels.append(y)

            original_filename = self.dataset.data_files[i]
            descr.append(original_filename.strip(self.dataset.extension))

        # Join features to create the dataset
        data = np.concatenate(data, axis=1)

        # Add label information
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=0)
        data = np.concatenate([data, labels], axis=0)
        data = np.transpose(data)

        # Store the features into a .csv file
        self.raw_data = data
        self.store_dataset(data, self.descr, self.feature_file)
        print("Raw dataset stored:", data.shape)

    def extract_features(self, x):
        features = []
        for j in range(self.n_joints):
            # Extract coordinates and compute L2 norm
            coord = x[:, j:j+3]

            if self.include_l2:
                l2_norms = np.linalg.norm(coord, axis=1)
                l2_norms = np.expand_dims(l2_norms, axis=1)
                coord = np.concatenate((coord, l2_norms), axis=1)

            # Split the array into time windows
            sub_arr = np.array_split(coord, self.n_windows, axis=0)

            # Extract features per each coordinate
            temp_features = []
            for arr in sub_arr:
                temp = []
                if "mean" in self.feature_names:
                    mean = np.mean(arr, axis=0)
                    temp.append(mean)
                if "absolute_harmonic_mean" in self.feature_names:
                    abs_harm_mean = hmean(np.abs(arr), axis=0)
                    temp.append(abs_harm_mean)
                if "std" in self.feature_names:
                    std = np.std(arr, axis=0)
                    temp.append(std)
                if "min" in self.feature_names:
                    maximum = np.max(arr, axis=0)
                    temp.append(maximum)
                if "max" in self.feature_names:
                    minimum = np.min(arr, axis=0)
                    temp.append(minimum)
                if "range" in self.feature_names:
                    tot_range = np.max(arr, axis=0) - np.min(arr, axis=0)
                    temp.append(tot_range)
                if "median" in self.feature_names:
                    median = np.median(arr, axis=0)
                    temp.append(median)
                if "median_absolute_dev" in self.feature_names:
                    median_abs_dev = median_abs_deviation(arr, axis=0)
                    temp.append(median_abs_dev)
                if "iq_range" in self.feature_names:
                    iq_range = iqr(arr, axis=0)
                    temp.append(iq_range)
                if "sum_of_area" in self.feature_names:
                    sum_of_area = np.sum(np.abs(arr), axis=0)
                    temp.append(sum_of_area)
                if "mean_energy" in self.feature_names:
                    mean_energy = np.mean(np.power(arr, 2), axis=0)
                    temp.append(mean_energy)
                if "skewness" in self.feature_names:
                    skewness = skew(arr, axis=0)
                    temp.append(skewness)
                if "kurtosis" in self.feature_names:
                    kurt = kurtosis(arr, axis=0)
                    temp.append(kurt)
                if "pearson_corr_coeff" in self.feature_names:
                    pearson = np.corrcoef(arr[:, :3], rowvar=False)
                    pearson = np.array([pearson[0, 1], pearson[0, 2], pearson[1, 2]])
                    temp.append(pearson)
                if "mean_velocity" in self.feature_names:
                    vel = np.diff(arr, axis=0) / self.dataset.dt
                    mean_vel = np.mean(vel, axis=0)
                    temp.append(mean_vel)

                # Join features for one time window
                temp = np.concatenate(temp)
                temp_features.append(temp)

            # Join features for one signal
            temp_features = np.concatenate(temp_features)
            features.append(temp_features)

        # Join features for every joint
        features = np.concatenate(features, axis=0)
        features = np.expand_dims(features, axis=1)

        return features

    def remove_nan(self):
        # Remove rows with NaN elements
        items = np.array_split(self.raw_data, self.raw_data.shape[0], axis=0)
        ind = [i for i in range(self.raw_data.shape[0]) if not np.any(np.isnan(items[i]))]

        items = [items[i] for i in ind]
        data = np.concatenate(items, axis=0)
        descr = [self.descr[i] for i in ind]
        print("After removing rows with NaN elements:", data.shape)

        # Store results
        self.preprocessed_data = data
        self.preprocessed_descr = descr
        self.store_dataset(data, descr, self.feature_file)
        print("Preprocessed dataset stored!")

    def store_dataset(self, data, descr, filename):
        # Store data
        headers = self.file_headers
        path = self.data_processing_dir + filename
        np.savetxt(path, data, delimiter=";", header=headers)

        # Store descriptors
        path = self.data_processing_dir + "descr_" + filename
        np.savetxt(path, descr, delimiter=";", fmt="%s")

    @staticmethod
    def load_data_matrix(working_dir, file_path, dtype="float", show_shape=False):
        data = np.loadtxt(fname=working_dir + file_path, delimiter=";", dtype=dtype)

        if show_shape:
            print("An array of shape", data.shape, "has been loaded")

        return data

    @staticmethod
    def read_feature_file(working_dir, feature_file, only_descriptors=False, group_dict=None, is_rehab=False):
        if is_rehab:
            feature_file = "rehab_" + feature_file
            prefix = "../IntelliRehabDS/"
        else:
            prefix = ""

        if not only_descriptors or group_dict is not None:
            addon = ""
            dtype = "float"
        else:
            addon = "descr_"
            dtype = "str"

        file_path = prefix + FeatureExtractor.data_processing_fold + addon + feature_file
        data_matrix = FeatureExtractor.load_data_matrix(working_dir=working_dir, file_path=file_path, dtype=dtype)
        dim = data_matrix.shape[0]

        # Select desired elements
        if not is_rehab and group_dict is not None:
            file_path = FeatureExtractor.data_processing_fold + "descr_" + feature_file
            descr = FeatureExtractor.load_data_matrix(working_dir=working_dir, file_path=file_path, dtype="str")
            elements = SkeletonDataset.find_elements(descr, group_dict)

            ind = [i for i in range(dim) if descr[i] in elements]
            data_matrix = data_matrix[ind, :]
            dim = data_matrix.shape[0]

        return data_matrix, dim

    @staticmethod
    def find_patient_indexes(working_dir, feature_file, patients, group_dict=None, is_rehab=False):
        data_descr, dim = FeatureExtractor.read_feature_file(working_dir=working_dir, feature_file=feature_file,
                                                             only_descriptors=True, is_rehab=is_rehab)

        # Select desired elements
        if not is_rehab and group_dict is not None:
            data_descr = SkeletonDataset.find_elements(data_descr, group_dict)
            dim = len(data_descr)

        ind = []
        for pt in patients:
            if not is_rehab:
                substr = "P" + f"{pt:03}"
                ind += [index for index in range(dim) if substr in data_descr[index]]
            else:
                substr = f"{pt:03}"
                ind += [index for index in range(dim) if str(data_descr[index]).startswith(substr)]

        return ind

    @staticmethod
    def normalize_data(x, mean=None, std=None):
        flag = False
        if mean is None or std is None:
            mean = np.mean(x, 0)
            std = np.std(x, 0)
            flag = True

        x = (x - mean) / std

        if not flag:
            return x
        else:
            return x, mean, std


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    # desired_classes1 = [8, 9] # NTU HAR binary
    # desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99] # NTU HAR multiclass
    desired_classes1 = [1, 2]  # IntelliRehabDS correctness
    # desired_classes1 = list(range(3, 12))  # IntelliRehabDS gesture

    feature_file1 = "hand_crafted_features_global.csv"
    # feature_file1 = "hand_crafted_features_global_15classes.csv"
    n_windows1 = 1
    include_l21 = False
    # selected_features1 = ["mean", "std", "mean_velocity"]
    selected_features1 = None
    is_rehab1 = True
    group_dict1 = {"C": 2, "R": 2} if not is_rehab1 else None

    # Define dataset instance
    # dataset1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1)
    dataset1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, maximum_length=200)

    # Build feature dataset
    feature_extractor1 = FeatureExtractor(working_dir=working_dir1, feature_file=feature_file1, dataset=dataset1,
                                          n_windows=n_windows1, include_l2=include_l21,
                                          selected_features=selected_features1, is_rehab=is_rehab1)
    feature_extractor1.build_feature_dataset()

    # Preprocess dataset
    feature_extractor1.remove_nan()

    # Load data
    data1, dim1 = FeatureExtractor.read_feature_file(working_dir=working_dir1, feature_file=feature_file1,
                                                     group_dict=group_dict1, is_rehab=is_rehab1)
