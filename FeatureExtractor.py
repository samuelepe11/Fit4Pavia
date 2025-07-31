# Import packages
import os

import numpy as np
import pandas as pd
from scipy.stats import hmean, median_abs_deviation, iqr, skew, kurtosis, mode

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
    bone_extremities = {"homerus_l": (4, 5), "homerus_r": (8, 9), "radius_l": (5, 6), "radius_r": (9, 10),
                        "femur_l": (12, 13), "femur_r": (16, 17), "tibia_l": (13, 14), "tibia_r": (17, 18),
                        "foot_l": (14, 15), "foot_r": (18, 19), "abdomen": (0, 1), "chest": (1, 20), "neck": (20, 2),
                        "head": (2, 3), "transverse_chest": (4, 8)}

    def __init__(self, working_dir, feature_file, dataset, n_windows, include_l2=True, selected_features=None,
                 is_rehab=False):
        if is_rehab:
            self.data_processing_fold = "../IntelliRehabDS/" + self.data_processing_fold
            feature_file = "rehab_" + feature_file if feature_file is not None else feature_file
        self.is_rehab = is_rehab

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

    def compute_subj_features_test(self):
        bone_extremities = {"homerus_l": (5, 6), "homerus_r": (2, 3), "radius_l": (6, 7), "radius_r": (3, 4),
                            "femur_l": (12, 13), "femur_r": (9, 10), "tibia_l": (13, 14), "tibia_r": (10, 11),
                            "transverse_chest": (2, 5)}
        list_pat = ["giovanna"]

        filedir = self.working_dir + "/../OpenCapData/"
        files = os.listdir(filedir)
        temp_feats = []
        for file in files:
            df = pd.read_csv(filedir + file, sep="\t", skiprows=4)
            df.drop(columns=["Unnamed: 0", "Unnamed: 1"], inplace=True)
            hip_c = np.array(df[["X8", "Y8", "Z8"]]).mean()
            hip_l = np.array(df[["X12", "Y12", "Z12"]]).mean()
            hip_r = np.array(df[["X9", "Y9", "Z9"]]).mean()
            hip_joints = [hip_c, hip_l, hip_r]

            knee_l = np.array(df[["X13", "Y13", "Z13"]]).mean()
            knee_r = np.array(df[["X10", "Y10", "Z10"]]).mean()
            ankle_l = np.array(df[["X14", "Y14", "Z14"]]).mean()
            ankle_r = np.array(df[["X11", "Y11", "Z11"]]).mean()
            foot_l = np.array(df[["X15", "Y15", "Z15"]]).mean()
            foot_r = np.array(df[["X18", "Y18", "Z18"]]).mean()
            lower_body_joints = [knee_l, knee_r, ankle_l, ankle_r, foot_l, foot_r]

            mean_lengths = {}
            for bone, (j1, j2) in bone_extremities.items():
                j1_cols = ["X" + str(j1), "Y" + str(j1), "Z" + str(j1)]
                j2_cols = ["X" + str(j2), "Y" + str(j2), "Z" + str(j2)]
                diffs = df[j2_cols].values - df[j1_cols].values
                lengths = np.linalg.norm(diffs, axis=1)
                mean_lengths[bone] = np.mean(lengths) * 100
            homerus = np.mean([mean_lengths["homerus_l"], mean_lengths["homerus_r"]])
            radius = np.mean([mean_lengths["radius_l"], mean_lengths["radius_r"]])
            femur = np.mean([mean_lengths["femur_l"], mean_lengths["femur_r"]])
            tibia = np.mean([mean_lengths["tibia_l"], mean_lengths["tibia_r"]])
            stature_h = 4.62 * homerus + 19.0
            stature_r = 3.78 * radius + 74.7
            stature_f = 2.71 * femur + 45.86
            stature_t = 3.29 * tibia + 47.34
            statures_all = [stature_f]

            temp_feats.append([file, hip_joints, lower_body_joints, statures_all, mean_lengths["transverse_chest"]])

        # Set data for hip and lower body joints distributions adjustment
        hips = [hip for _, hip, _, _, _ in temp_feats]
        m_hip = np.mean(hips, axis=0)
        s_hip = np.std(hips, axis=0)
        population_m_hip = [(0.9918 + 1.1906) / 2, (0.9738 + 1.1673) / 2, (0.9724 + 1.1754) / 2]
        population_s_hip = np.sqrt([(0.1154 ** 2 + 0.1489 ** 2) / 2 + (0.9918 - 1.1906) ** 2 / 4,
                                    (0.1172 ** 2 + 0.1529 ** 2) / 2 + (0.9738 - 1.1673) ** 2 / 4,
                                    (0.1155 ** 2 + 0.1481 ** 2) / 2 + (0.9724 - 1.1754) ** 2 / 4])

        lbs = [lb for _, _, lb, _, _ in temp_feats]
        m_lb = np.mean(lbs, axis=0)
        s_lb = np.std(lbs, axis=0)
        population_m_lb = [(0.8130 + 1.0458) / 2, (0.8486 + 1.0804) / 2, (0.7400 + 0.9787) / 2,
                           (0.7607 + 1.0120) / 2, (0.7054 + 0.9499) / 2, (0.7217 + 0.9827) / 2]
        population_s_lb = np.sqrt([(0.1415 ** 2 + 0.1602 ** 2) / 2 + (0.8130 - 1.0458) ** 2 / 4,
                                   (0.1160 ** 2 + 0.1576 ** 2) / 2 + (0.8486 - 1.0804) ** 2 / 4,
                                   (0.1224 ** 2 + 0.1615 ** 2) / 2 + (0.7400 - 0.9787) ** 2 / 4,
                                   (0.1172 ** 2 + 0.1733 ** 2) / 2 + (0.7607 - 1.0120) ** 2 / 4,
                                   (0.1240 ** 2 + 0.1616 ** 2) / 2 + (0.7054 - 0.9499) ** 2 / 4,
                                   (0.1180 ** 2 + 0.1751 ** 2) / 2 + (0.7217 - 0.9827) ** 2 / 4])

        # Infer gender
        pooled_gender_list = []
        for _, hip, lb, _, _ in temp_feats:
            # Infer gender
            hip = (hip - m_hip) / s_hip * population_s_hip + population_m_hip
            hip = np.mean(hip)
            lb = (lb - m_lb) / s_lb * population_s_lb + population_m_lb
            lower_body = np.mean(lb)
            p_female = 1 / (1 + np.exp(66.2 * hip - 75.0 * lower_body - 4.41))
            gender = int(p_female >= 0.5)
            pooled_gender_list.append(gender)

        subj_gender_list = []
        for subj in list_pat:
            subj_genders = np.array([pooled_gender_list[i] for i in range(len(pooled_gender_list)) if subj in
                                     temp_feats[i][0]])
            subj_gender_list.append(int(mode(subj_genders).mode))

        subj_feat_rows = []
        for descr_i, _, _, statures_all, transverse_chest_width in temp_feats:
            # Retrieve gender
            for i, subj in enumerate(list_pat):
                if subj in descr_i:
                    subj_ind = i
                    break
            gender = subj_gender_list[subj_ind]

            # Infer stature
            stature_threshold = 150 if gender else 160
            statures = [s for s in statures_all if s > stature_threshold]
            statures = statures if len(statures) > 0 else statures_all
            stature = np.mean(statures)

            # Infer weight
            if gender:
                weight_hv = 0.5 * stature - 30.563
                weight_tcb = 3.407 * transverse_chest_width - 35.686
                weight_threshold = 45
            else:
                weight_hv = 0.665 * stature - 54.477
                weight_tcb = 2.489 * transverse_chest_width - 9.742
                weight_threshold = 55
            weights_all = [weight_hv]
            weights = [w for w in weights_all if w > weight_threshold]
            weights = weights if len(weights) > 0 else weights_all
            weight = np.mean(weights)

            subj_feat_rows.append([stature, weight])

        # Average features across different items
        all_subj_feats = []
        for j, subj in enumerate(list_pat):
            subj_feats = np.array([subj_feat_rows[i] for i in range(len(subj_feat_rows)) if subj in temp_feats[i][0]])
            all_subj_feats.append([subj, subj_gender_list[j]] + list(np.mean(subj_feats[:, 0:], axis=0)))

            print("Subject", subj + ":", all_subj_feats[-1])

    def compute_subj_features(self, subj_filename, group_dict=None):
        features_dataset, dim = self.read_feature_file(self.working_dir, self.feature_file, only_descriptors=False,
                                                       is_rehab=self.is_rehab)
        descr, _ = self.read_feature_file(self.working_dir, self.feature_file, only_descriptors=True,
                                          is_rehab=self.is_rehab)
        subj_filename = subj_filename if not self.is_rehab else "rehab_" + subj_filename

        selected_descr = [d for d in descr if self.get_label_from_name(d) in self.dataset.classes]
        if group_dict is not None:
            selected_descr = SkeletonDataset.find_elements(selected_descr, group_dict)
        features_dataset = [feat for i, feat in enumerate(features_dataset) if descr[i] in selected_descr]
        dim = len(features_dataset)
        descr = selected_descr

        temp_feats = []
        for i, row in enumerate(features_dataset):
            # Compute body characteristics
            hip_c = np.mean(row[0:3])
            hip_l = np.mean(row[108:111])
            hip_r = np.mean(row[144:147])
            hip_joints = [hip_c, hip_l, hip_r]

            knee_l = np.mean(row[117:120])
            knee_r = np.mean(row[153:156])
            ankle_l = np.mean(row[126:129])
            ankle_r = np.mean(row[162:165])
            foot_l = np.mean(row[135:138])
            foot_r = np.mean(row[171:174])
            lower_body_joints = [knee_l, knee_r, ankle_l, ankle_r, foot_l, foot_r]

            mean_lengths = {}
            skel_data = self.dataset.__getitem__(i)[0]
            for bone, (j1, j2) in self.bone_extremities.items():
                j1_indices = [3 * j1, 3 * j1 + 1, 3 * j1 + 2]
                j2_indices = [3 * j2, 3 * j2 + 1, 3 * j2 + 2]
                diffs = skel_data[:, j2_indices] - skel_data[:, j1_indices]
                lengths = np.linalg.norm(diffs, axis=1)
                mean_lengths[bone] = np.mean(lengths) * 100
            homerus = np.mean([mean_lengths["homerus_l"], mean_lengths["homerus_r"]])
            radius = np.mean([mean_lengths["radius_l"], mean_lengths["radius_r"]])
            femur = np.mean([mean_lengths["femur_l"], mean_lengths["femur_r"]])
            tibia = np.mean([mean_lengths["tibia_l"], mean_lengths["tibia_r"]])
            stature_h = 4.62 * homerus + 19.0
            stature_r = 3.78 * radius + 74.7
            stature_f = 2.71 * femur + 45.86
            stature_t = 3.29 * tibia + 47.34

            foot = np.mean([mean_lengths["foot_l"], mean_lengths["foot_r"]])
            stature_bone_chain = (foot + tibia + femur + mean_lengths["abdomen"] + mean_lengths["chest"] +
                                  mean_lengths["neck"] + mean_lengths["head"])
            statures_all = [stature_h, stature_r, stature_f, stature_t, stature_bone_chain]

            temp_feats.append([descr[i], hip_joints, lower_body_joints, statures_all, mean_lengths["transverse_chest"]])

        # Set data for hip and lower body joints distributions adjustment
        hips = [hip for _, hip, _, _, _ in temp_feats]
        m_hip = np.mean(hips, axis=0)
        s_hip = np.std(hips, axis=0)
        population_m_hip = [(0.9918 + 1.1906) / 2, (0.9738 + 1.1673) / 2, (0.9724 + 1.1754) / 2]
        population_s_hip = np.sqrt([(0.1154 ** 2 + 0.1489 ** 2) / 2 + (0.9918 - 1.1906) ** 2 / 4,
                                    (0.1172 ** 2 + 0.1529 ** 2) / 2 + (0.9738 - 1.1673) ** 2 / 4,
                                    (0.1155 ** 2 + 0.1481 ** 2) / 2 + (0.9724 - 1.1754) ** 2 / 4])

        lbs = [lb for _, _, lb, _, _ in temp_feats]
        m_lb = np.mean(lbs, axis=0)
        s_lb = np.std(lbs, axis=0)
        population_m_lb = [(0.8130 + 1.0458) / 2, (0.8486 + 1.0804) / 2, (0.7400 + 0.9787) / 2,
                           (0.7607 + 1.0120) / 2, (0.7054 + 0.9499) / 2, (0.7217 + 0.9827) / 2]
        population_s_lb = np.sqrt([(0.1415 ** 2 + 0.1602 ** 2) / 2 + (0.8130 - 1.0458) ** 2 / 4,
                                   (0.1160 ** 2 + 0.1576 ** 2) / 2 + (0.8486 - 1.0804) ** 2 / 4,
                                   (0.1224 ** 2 + 0.1615 ** 2) / 2 + (0.7400 - 0.9787) ** 2 / 4,
                                   (0.1172 ** 2 + 0.1733 ** 2) / 2 + (0.7607 - 1.0120) ** 2 / 4,
                                   (0.1240 ** 2 + 0.1616 ** 2) / 2 + (0.7054 - 0.9499) ** 2 / 4,
                                   (0.1180 ** 2 + 0.1751 ** 2) / 2 + (0.7217 - 0.9827) ** 2 / 4])

        # Infer gender
        pooled_gender_list = []
        for _, hip, lb, _, _ in temp_feats:
            # Infer gender
            hip = (hip - m_hip) / s_hip * population_s_hip + population_m_hip
            hip = np.mean(hip)
            lb = (lb - m_lb) / s_lb * population_s_lb + population_m_lb
            lower_body = np.mean(lb)
            p_female = 1 / (1 + np.exp(66.2 * hip - 75.0 * lower_body - 4.41))
            gender = int(p_female >= 0.5)
            pooled_gender_list.append(gender)

        subj_gender_list = []
        k = "P" if not self.is_rehab else 0
        for subj in self.dataset.list_pat:
            subj_items = self.dataset.find_elements(descr, {k: subj})
            subj_genders = np.array([pooled_gender_list[i] for i in range(dim) if descr[i] in subj_items])
            subj_gender_list.append(int(mode(subj_genders).mode))

        subj_feat_rows = []
        for descr_i, _, _, statures_all, transverse_chest_width in temp_feats:
            # Retrieve gender
            if not self.is_rehab:
                subj_ind = int(descr_i[9:12]) - 1
                gender = subj_gender_list[subj_ind]
            else:
                subj_ind = self.dataset.list_pat.index(int(descr_i[:3]))
                gender = self.dataset.genders[subj_ind]
                gender = gender if not np.isnan(gender) else subj_gender_list[subj_ind]

            # Infer stature
            stature_threshold = 150 if gender else 160
            statures = [s for s in statures_all if s > stature_threshold]
            statures = statures if len(statures) > 0 else statures_all
            stature = np.mean(statures)

            # Infer weight
            if gender:
                weight_hv = 0.5 * stature - 30.563
                weight_tcb = 3.407 * transverse_chest_width - 35.686
                weight_threshold = 45
            else:
                weight_hv = 0.665 * stature - 54.477
                weight_tcb = 2.489 * transverse_chest_width - 9.742
                weight_threshold = 55
            weights_all = [weight_hv, weight_tcb]
            weights = [w for w in weights_all if w > weight_threshold]
            weights = weights if len(weights) > 0 else weights_all
            weight = np.mean(weights)

            subj_feat_rows.append([stature, weight])

        # Average features across different items
        all_subj_feats = []
        for j, subj in enumerate(self.dataset.list_pat):
            subj_items = self.dataset.find_elements(descr, {k: subj})
            subj_feats = np.array([subj_feat_rows[i] for i in range(dim) if descr[i] in subj_items])
            ind = subj - 1 if not self.is_rehab else j
            if not self.is_rehab:
                gender = subj_gender_list[j]
            else:
                gender = self.dataset.genders[j]
                gender = gender if not np.isnan(gender) else subj_gender_list[j]
            subj_invariants = [subj, gender]
            if self.is_rehab:
                age = self.dataset.ages[ind]
                age = age if not np.isnan(age) else np.nanmedian(self.dataset.ages)
                subj_invariants += [age]
            all_subj_feats.append(subj_invariants + list(np.mean(subj_feats[:, 0:], axis=0)))

        headers = ["subj_id", "gender", "stature", "weight"]
        if self.is_rehab:
            headers.insert(2, "age")
        self.file_headers = ";".join(headers)
        data = np.array(all_subj_feats)
        self.store_dataset(data, None, subj_filename)
        print("Subject features stored:", data.shape)

        if self.is_rehab:
            gender_acc = np.mean([self.dataset.genders[i] == subj_gender_list[i]
                                  for i in range(len(self.dataset.list_pat)) if not np.isnan(self.dataset.genders[i])])
            print("Gender prediction accuracy = " + str(np.round(gender_acc * 100, 2)) + "%")

    def get_label_from_name(self, name):
        label = int(name[-3:]) if not self.is_rehab else int(name.split("_")[4])
        return label

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

    def store_dataset(self, data, descr, filename, is_variance_analysis=False):
        dir_path = self.data_processing_dir
        if is_variance_analysis:
            dir_path += "variance_analysis/"
            if "variance_analysis" not in os.listdir(self.data_processing_dir):
                os.mkdir(dir_path)

        if self.is_rehab:
            filename = "rehab_" + filename

        # Store data
        headers = self.file_headers
        path = dir_path + filename
        np.savetxt(path, data, delimiter=";", header=headers)

        # Store descriptors
        if descr is not None:
            path = dir_path + "descr_" + filename
            np.savetxt(path, descr, delimiter=";", fmt="%s")

    @staticmethod
    def load_data_matrix(working_dir, file_path, dtype="float", show_shape=False):
        data = np.loadtxt(fname=working_dir + file_path, delimiter=";", dtype=dtype)

        if show_shape:
            print("An array of shape", data.shape, "has been loaded")

        return data

    @staticmethod
    def read_feature_file(working_dir, feature_file, only_descriptors=False, group_dict=None, is_rehab=False,
                          variance_analysis_class_n=None):
        if is_rehab:
            feature_file = "rehab_" + feature_file if not feature_file.startswith("rehab_") else feature_file
            prefix = "../IntelliRehabDS/"
        else:
            prefix = ""

        if not only_descriptors or group_dict is not None:
            addon = ""
            dtype = "float"
        else:
            addon = "descr_"
            dtype = "str"

        if variance_analysis_class_n is not None:
            addon1 = "variance_analysis/"
            feature_file += "_" + str(variance_analysis_class_n) + "classes.csv"
        else:
            addon1 = ""

        file_path = prefix + FeatureExtractor.data_processing_fold + addon1 + addon + feature_file
        data_matrix = FeatureExtractor.load_data_matrix(working_dir=working_dir, file_path=file_path, dtype=dtype)
        dim = data_matrix.shape[0]

        # Select desired elements
        if not is_rehab and group_dict is not None:
            file_path = FeatureExtractor.data_processing_fold + addon1 + "descr_" + feature_file
            descr = FeatureExtractor.load_data_matrix(working_dir=working_dir, file_path=file_path, dtype="str")
            elements = SkeletonDataset.find_elements(descr, group_dict)

            ind = [i for i in range(dim) if descr[i] in elements]
            data_matrix = data_matrix[ind, :]
            dim = data_matrix.shape[0]

            if only_descriptors:
                return elements, dim

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
    # desired_classes1 = [8, 9]  # NTU HAR binary
    # desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]  # NTU HAR multiclass
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
    # dataset1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, group_dict=group_dict1)
    dataset1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, maximum_length=200)

    # Build feature dataset
    feature_extractor1 = FeatureExtractor(working_dir=working_dir1, feature_file=feature_file1, dataset=dataset1,
                                          n_windows=n_windows1, include_l2=include_l21,
                                          selected_features=selected_features1, is_rehab=is_rehab1)
    # feature_extractor1.build_feature_dataset()

    # Preprocess dataset
    # feature_extractor1.remove_nan()

    # Load data
    # data1, dim1 = FeatureExtractor.read_feature_file(working_dir=working_dir1, feature_file=feature_file1,
    #                                                  group_dict=group_dict1, is_rehab=is_rehab1)

    # Build subject-specific feature dataset
    subj_filename1 = "subject_features.csv"
    # feature_extractor1.compute_subj_features_test()
    feature_extractor1.compute_subj_features(subj_filename1, group_dict1)
