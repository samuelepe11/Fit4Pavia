# Import packages
from enum import Enum


# Class
class MLAlgorithmType(Enum):
    KNN = "k-NN"
    KNN_DTW = "Dynamic Time Warping k-NN"
    SVM = "Kernel SVM"
    RF = "Random Forest"
    AB = "AdaBoost"
    MLP = "Multi-layer perceptron"
