# Import packages
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.special import expit, softmax
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

from SetType import SetType
from MLAlgorithmType import MLAlgorithmType
from FeatureExtractor import FeatureExtractor
from StatsHolder import StatsHolder
from Trainer import Trainer
from SkeletonDataset import SkeletonDataset
from RehabSkeletonDataset import RehabSkeletonDataset


# Class
class SimpleClassifierTrainer(Trainer):

    def __init__(self, ml_algorithm, working_dir, folder_name, train_data, test_data, normalize_data=False,
                 desired_classes=None, is_rehab=False):
        super().__init__(working_dir, folder_name, train_data, test_data, is_rehab)

        # Initialize attributes
        self.ml_algorithm = ml_algorithm
        self.desired_classes = desired_classes
        if ml_algorithm == MLAlgorithmType.SVM:
            if not is_rehab:
                # params = [0.5, "rbf"]  # Regularization parameter and kernel type (apply default settings for each kernel)
                params = [0.75, "rbf"]  # Multiclass case
            else:
                params = [10, "rbf"]
            self.model = SVC(C=params[0], kernel=params[1])
        elif ml_algorithm == MLAlgorithmType.RF:
            if not is_rehab:
                params = [100, "gini"]  # Number of trees and impurity measure
            else:
                params = [50, "gini"]
            self.model = RandomForestClassifier(n_estimators=params[0], criterion=params[1])
        elif ml_algorithm == MLAlgorithmType.AB:
            if not is_rehab:
                # params = [DecisionTreeClassifier(max_depth=1), 100, 1]  # Base classifier, number of estimators and LR
                params = [DecisionTreeClassifier(max_depth=3), 200, 1]  # Multiclass case
            else:
                params = [DecisionTreeClassifier(max_depth=3), 200, 1]
            self.model = AdaBoostClassifier(estimator=params[0], n_estimators=params[1], learning_rate=params[2],
                                            algorithm="SAMME")
        elif ml_algorithm == MLAlgorithmType.MLP:
            if not is_rehab:
                # params = [(64,), 0.01]  # Hidden layer sizes and initial LR
                params = [(128,), 0.001]  # Multiclass case
            else:
                params = [(64,), 0.001]
            self.model = MLPClassifier(hidden_layer_sizes=params[0], learning_rate_init=params[1])
        elif ml_algorithm == MLAlgorithmType.KNN:
            if not is_rehab:
                params = [5]  # Number of neighbors
            else:
                params = [3]
            self.model = KNeighborsClassifier(n_neighbors=params[0])
        else:
            # Dynamic Time Warping KNN
            if not is_rehab:
                # params = [5]  # Number of neighbors
                params = [7]  # Multiclass case
            else:
                print("TODO")
            self.model = KNeighborsTimeSeriesClassifier(n_neighbors=params[0], metric="dtw")

            # Adjust the data
            self.train_data = self.train_data.min_max_scale()
            self.test_data = self.test_data.min_max_scale()

        self.train_losses.append(float("inf"))

        self.normalize_data = normalize_data
        self.x_mean = None
        self.x_std = None

    def train(self, filename=None):
        data = self.train_data

        if self.ml_algorithm != MLAlgorithmType.KNN_DTW:
            np.random.shuffle(data)
            x = data[:, :-1]
            if self.normalize_data:
                x, self.x_mean, self.x_std = FeatureExtractor.normalize_data(x)
            y = data[:, -1]
        else:
            x, y = data
            combined_data = list(zip(x, y))
            random.shuffle(combined_data)
            x, y = zip(*combined_data)

            x = to_time_series_dataset(x)
            y = np.asarray(y)
        y = y.astype(int)
        self.model.fit(x, y)

        acc = self.model.score(x, y)
        loss = 1 - acc
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        SimpleClassifierTrainer.save_model(self, filename, use_keras=False)

    def test(self, set_type=SetType.TRAINING, show_cm=False, avoid_eval=False, assess_calibration=False,
             return_only_preds=False, return_also_preds=False, is_rehab=False, ext_test_data=None, ext_test_name=None,
             for_generation=False):
        if set_type == SetType.TRAINING:
            data = self.train_data
        else:
            data = self.test_data

        if self.ml_algorithm != MLAlgorithmType.KNN_DTW:
            x = data[:, :-1]
            if self.normalize_data:
                x = FeatureExtractor.normalize_data(x, self.x_mean, self.x_std)
            y = data[:, -1]
        else:
            x, y = data
            x = to_time_series_dataset(x)
            y = np.asarray(y)
        y = y.astype(int)
        prediction = self.model.predict(x)

        # Accuracy evaluation
        acc = self.model.score(x, y)
        loss = 1 - acc

        # Store values for Confusion Matrix calculation
        y_true = y
        y_pred = prediction
        if return_only_preds:
            try:
                y_prob = self.model.predict_proba(x)
            except AttributeError:
                decision_scores = self.model.decision_function(x)
                if len(decision_scores.shape) == 1 or decision_scores.shape[1] == 2:
                    y_prob = expit(decision_scores)
                    y_prob = np.stack([1 - y_prob, y_prob], -1)
                    if len(decision_scores.shape) > 1 and decision_scores.shape[1] == 2:
                        y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)
                else:
                    y_prob = softmax(decision_scores, axis=1)
            y_prob = np.array([y_prob[k, int(y_pred[k])] for k in range(len(y_pred))])
            return y_true, y_pred, y_prob

        # Confusion matrix definition
        cm = Trainer.compute_binary_confusion_matrix(prediction, y, classes=range(len(self.desired_classes)))
        tp = np.float64(cm[0])
        tn = np.float64(cm[1])
        fp = np.float64(cm[2])
        fn = np.float64(cm[3])
        stats_holder = StatsHolder(loss, acc, tp, tn, fp, fn)

        # Compute multiclass confusion matrix
        cm_name = set_type.value + "_cm"
        if show_cm:
            img_path = self.results_dir + self.model_name + "/" + cm_name + ".png"
        else:
            img_path = None
        self.__dict__[cm_name] = Trainer.compute_multiclass_confusion_matrix(y_true, y_pred, self.desired_classes,
                                                                             img_path)

        return stats_holder

    def show_model(self):
        print("ML MODEL:")
        print(self.model)

    def find_data_files(self, data, feature_file=None, is_rehab=False):
        gd = {"C": 2, "R": 2} if not is_rehab else 200
        all_data, _ = FeatureExtractor.read_feature_file(self.working_dir, feature_file, only_descriptors=False,
                                                         group_dict=gd, is_rehab=is_rehab)
        all_data_descr, _ = FeatureExtractor.read_feature_file(self.working_dir, feature_file, only_descriptors=True,
                                                               is_rehab=is_rehab)
        if not is_rehab:
            all_data_descr = SkeletonDataset.find_elements(all_data_descr, group_dict=gd)
        x = data
        data_files = []
        for i in range(x.shape[0]):
            xi = x[i, :]
            for j in range(len(all_data)):
                xj = all_data[j, :]
                if (xi == xj).all():
                    data_files.append(all_data_descr[j])
        return data_files


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 111099
    random.seed(seed)
    np.random.seed(seed)

    # Define variables
    working_dir1 = "./../"
    # desired_classes1 = [8, 9] # NTU HAR binary
    desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]  # NTU HAR multiclass
    # desired_classes1 = [1, 2]  # IntelliRehabDS correctness
    # desired_classes1 = list(range(3, 12))  # IntelliRehabDS gesture

    is_rehab1 = False
    group_dict1 = {"C": 2, "R": 2} if not is_rehab1 else None

    # Read the data
    data_file = "hand_crafted_features_global.csv"
    data_file = "hand_crafted_features_global_15classes.csv"
    data_matrix, dim = FeatureExtractor.read_feature_file(working_dir=working_dir1, feature_file=data_file,
                                                          group_dict=group_dict1, is_rehab=is_rehab1)

    # Divide the dataset for simple ML models
    train_perc = 0.7
    num_train_pt = round(dim * train_perc)
    ind_train = random.sample(range(dim), num_train_pt)
    train_data1 = data_matrix[ind_train, :]
    ind_test = [i for i in range(dim) if i not in ind_train]
    test_data1 = data_matrix[ind_test, :]

    # Define the model
    folder_name1 = "tests"
    model_name1 = "prova"
    ml_algorithm1 = MLAlgorithmType.SVM

    # Define the data for DTW KNN
    if ml_algorithm1 == MLAlgorithmType.KNN_DTW:
        '''train_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                      group_dict={"C": 2, "R": 2}, data_perc=train_perc)
        test_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                     data_names=train_data1.remaining_instances)'''
        train_data1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                           data_perc=train_perc,
                                           divide_pt=True, maximum_length=200)
        test_data1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                          data_names=train_data1.remaining_instances)

    normalize_data1 = True
    trainer1 = SimpleClassifierTrainer(ml_algorithm=ml_algorithm1, working_dir=working_dir1, folder_name=folder_name1,
                                       train_data=train_data1, test_data=test_data1, normalize_data=normalize_data1,
                                       desired_classes=desired_classes1, is_rehab=is_rehab1)

    # Train the model
    trainer1.train(model_name1)
    trainer1.summarize_performance()

    # Load trained model
    trainer1 = Trainer.load_model(working_dir=working_dir1, folder_name=folder_name1, model_name=model_name1,
                                  is_rehab=is_rehab1)
    trainer1.summarize_performance()
