# Import packages
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

from SetType import SetType
from MLAlgorithmType import MLAlgorithmType
from FeatureExtractor import FeatureExtractor
from StatsHolder import StatsHolder
from Trainer import Trainer
from SkeletonDataset import SkeletonDataset


# Class
class SimpleClassifierTrainer(Trainer):

    def __init__(self, ml_algorithm, working_dir, folder_name, train_data, test_data):
        super().__init__(working_dir, folder_name, train_data, test_data)

        # Initialize attributes
        self.ml_algorithm = ml_algorithm
        if ml_algorithm == MLAlgorithmType.SVM:
            params = [0.5, "rbf"]  # Regularization parameter and kernel type (apply default settings for each kernel)
            self.model = SVC(C=params[0], kernel=params[1])
        elif ml_algorithm == MLAlgorithmType.RF:
            params = [100, "gini"]  # Number of tress and impurity measure
            self.model = RandomForestClassifier(n_estimators=params[0], criterion=params[1])
        elif ml_algorithm == MLAlgorithmType.AB:
            params = [100]  # Number of estimators
            self.model = AdaBoostClassifier(n_estimators=params[0], algorithm="SAMME")
        elif ml_algorithm == MLAlgorithmType.MLP:
            params = [(64,), 0.01]  # Hidden layer sizes and initial learning rate
            self.model = MLPClassifier(hidden_layer_sizes=params[0], learning_rate_init=params[1])
        elif ml_algorithm == MLAlgorithmType.KNN:
            params = [5]  # Number of neighbors
            self.model = KNeighborsClassifier(n_neighbors=params[0])
        else:
            # Dynamic Time Warping KNN
            params = [5]  # Number of neighbors
            self.model = KNeighborsTimeSeriesClassifier(n_neighbors=params[0], metric="dtw")

            # Adjust the data
            self.train_data = self.train_data.min_max_scale()
            self.test_data = self.test_data.min_max_scale()

        self.train_losses.append(float("inf"))

    def train(self, filename=None):
        data = self.train_data

        if self.ml_algorithm != MLAlgorithmType.KNN_DTW:
            np.random.shuffle(data)
            x = data[:, :-1]
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

        loss = 1 - self.model.score(x, y)
        self.train_losses.append(loss)
        SimpleClassifierTrainer.save_model(self, filename, use_keras=False)

    def test(self, set_type=SetType.TRAINING):
        if set_type == SetType.TRAINING:
            data = self.train_data
        else:
            data = self.test_data

        if self.ml_algorithm != MLAlgorithmType.KNN_DTW:
            x = data[:, :-1]
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

        # Confusion matrix definition
        cm = SimpleClassifierTrainer.compute_confusion_matrix(prediction, y)
        tp = np.float64(cm[0])
        tn = np.float64(cm[1])
        fp = np.float64(cm[2])
        fn = np.float64(cm[3])

        stats_holder = StatsHolder(loss, acc, tp, tn, fp, fn)
        return stats_holder


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 111099
    random.seed(seed)

    # Define variables
    working_dir1 = "./../"
    desired_classes1 = [8, 9]

    # Read the data
    data_file = "hand_crafted_features_global.csv"
    data_matrix, dim = FeatureExtractor.read_feature_file(working_dir=working_dir1, feature_file=data_file)

    # Divide the dataset for simple ML models
    train_perc = 0.7
    num_train_pt = round(dim * train_perc)
    ind_train = random.sample(range(dim), num_train_pt)
    train_data1 = data_matrix[ind_train, :]
    ind_test = [i for i in range(dim) if i not in ind_train]
    test_data1 = data_matrix[ind_test, :]

    # Define the model
    folder_name1 = "tests"
    folder_name1 = "patientVSrandom_division_knn_dtw/sit_patient_division"
    model_name1 = "test_refac"
    model_name1 = "trial_0"
    ml_algorithm1 = MLAlgorithmType.KNN_DTW

    # Define the data for DTW KNN
    if ml_algorithm1 == MLAlgorithmType.KNN_DTW:
        train_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                      group_dict={"C": 2, "R": 2}, data_perc=train_perc)
        test_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                     data_names=train_data1.remaining_instances)

    trainer1 = SimpleClassifierTrainer(ml_algorithm=ml_algorithm1, working_dir=working_dir1, folder_name=folder_name1,
                                       train_data=train_data1, test_data=test_data1)

    # Train the model
    trainer1.train(model_name1)
    trainer1.summarize_performance()

    # Load trained model
    # trainer1 = Trainer.load_model(working_dir=working_dir1, folder_name=folder_name1, model_name=model_name1)
    # trainer1.summarize_performance()
