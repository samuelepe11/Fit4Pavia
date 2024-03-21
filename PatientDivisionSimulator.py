# Import packages
import random

from Simulator import Simulator
from SkeletonDataset import SkeletonDataset
from NetType import NetType
from FeatureExtractor import FeatureExtractor
from MLAlgorithmType import MLAlgorithmType


# Class
class PatientDivisionSimulator(Simulator):

    def __init__(self, desired_classes, n_rep, simulator_name, working_dir, folder_name, model_type, train_perc,
                 data_group_dict=None, train_epochs=None, train_lr=None, feature_file=None):
        super().__init__(desired_classes, n_rep, simulator_name, working_dir, folder_name, model_type, train_perc,
                         data_group_dict, train_epochs, train_lr, feature_file)

    def divide_dataset(self):
        # Perform a division per patient
        if self.feature_file is None or self.model_type == MLAlgorithmType.KNN_DTW:
            # Read skeleton data
            train_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                         group_dict=self.data_group_dict, data_perc=self.train_perc, divide_pt=True)
            test_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                        data_names=train_data.remaining_instances)
        else:
            # Read feature data
            data_matrix, dim = FeatureExtractor.read_feature_file(working_dir=self.working_dir,
                                                                  feature_file=self.feature_file,
                                                                  group_dict=self.data_group_dict)

            # Divide the dataset
            num_train_pt = round(SkeletonDataset.n_patients * self.train_perc)
            selected_patients = random.sample(range(1, SkeletonDataset.n_patients + 1), num_train_pt)
            ind = FeatureExtractor.find_patient_indexes(working_dir=self.working_dir, feature_file=self.feature_file,
                                                        patients=selected_patients, group_dict=self.data_group_dict)

            train_data, test_data = Simulator.divide_features(data_matrix, dim, ind)

        return train_data, test_data


# Main
if __name__ == "__main__":
    # Define variables
    seed1 = 111099
    working_dir1 = "./../"
    desired_classes1 = [8, 9]
    # desired_classes1 = [69, 70]

    data_group_dict1 = {"C": 2, "R": 2}
    model_type1 = NetType.CONV2D_NO_HYBRID
    # model_type1 = MLAlgorithmType.SVM
    train_perc1 = 0.7
    n_rep1 = 100
    train_epochs1 = 300
    train_lr1 = 0.01
    folder_name1 = "patientVSrandom_division_conv"
    simulator_name1 = "sit_patient_division"

    feature_file1 = "hand_crafted_features_global.csv"

    # Initialize the simulator
    simulator1 = PatientDivisionSimulator(desired_classes=desired_classes1, n_rep=n_rep1,
                                          simulator_name=simulator_name1, working_dir=working_dir1,
                                          folder_name=folder_name1, data_group_dict=data_group_dict1,
                                          model_type=model_type1, train_perc=train_perc1, train_epochs=train_epochs1,
                                          train_lr=train_lr1)
    # simulator1 = PatientDivisionSimulator(desired_classes=desired_classes1, n_rep=n_rep1,
    #                                       simulator_name=simulator_name1, working_dir=working_dir1,
    #                                       folder_name=folder_name1, data_group_dict=data_group_dict1,
    #                                       model_type=model_type1, train_perc=train_perc1, feature_file=feature_file1)

    # Load simulator
    # simulator1 = Simulator.load_simulator(working_dir1, folder_name1, simulator_name1)

    # Run simulation
    simulator1.run_simulation(seed1)
    simulator1.log_simulation_results()

    # Reload simulation results (in case of substantial modifications to the computed statistics)
    # simulator1.reload_simulation_results()

    # Assess and store simulator
    simulator1.assess_simulation(ci_alpha=0.05)
    Simulator.save_simulator(simulator1, simulator_name1)
