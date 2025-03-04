# Import packages
import random
import numpy as np

from Simulator import Simulator
from SkeletonDataset import SkeletonDataset
from RehabSkeletonDataset import RehabSkeletonDataset
from NetType import NetType
from FeatureExtractor import FeatureExtractor
from MLAlgorithmType import MLAlgorithmType


# Class
class PatientDivisionSimulator(Simulator):

    def __init__(self, desired_classes, n_rep, simulator_name, working_dir, folder_name, model_type, train_perc,
                 data_group_dict=None, train_epochs=None, train_lr=None, feature_file=None, normalize_data=False,
                 use_cuda=True, is_rehab=False):
        super().__init__(desired_classes, n_rep, simulator_name, working_dir, folder_name, model_type, train_perc,
                         data_group_dict, train_epochs, train_lr, feature_file, normalize_data, use_cuda, is_rehab)

    def divide_dataset(self):
        # Perform a division per patient
        if self.feature_file is None or self.model_type == MLAlgorithmType.KNN_DTW:
            # Read skeleton data
            if not self.is_rehab:
                train_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                             group_dict=self.data_group_dict, data_perc=self.train_perc, divide_pt=True)
                test_data = SkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                            data_names=train_data.remaining_instances)
            else:
                train_data = RehabSkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                                  data_perc=self.train_perc, divide_pt=True,
                                                  maximum_length=self.data_group_dict)
                test_data = RehabSkeletonDataset(working_dir=self.working_dir, desired_classes=self.desired_classes,
                                                 data_names=train_data.remaining_instances)
        else:
            # Read feature data
            data_matrix, dim = FeatureExtractor.read_feature_file(working_dir=self.working_dir,
                                                                  feature_file=self.feature_file,
                                                                  group_dict=self.data_group_dict,
                                                                  is_rehab=self.is_rehab)

            # Divide the dataset
            if not self.is_rehab:
                if np.all([c <= 60 for c in self.desired_classes]):
                    n_patients = 40
                else:
                    n_patients = 106
            else:
                n_patients = len(RehabSkeletonDataset.list_pat)
            num_train_pt = round(n_patients * self.train_perc)
            selected_patients = random.sample(range(1, n_patients + 1), num_train_pt)
            if self.is_rehab:
                selected_patients = np.array([RehabSkeletonDataset.list_pat[ind_pt - 1] for ind_pt in selected_patients])
            ind = FeatureExtractor.find_patient_indexes(working_dir=self.working_dir, feature_file=self.feature_file,
                                                        patients=selected_patients, group_dict=self.data_group_dict,
                                                        is_rehab=is_rehab1)

            train_data, test_data = Simulator.divide_features(data_matrix, dim, ind)

        return train_data, test_data


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
    simulator_name1 = "patient_division"
    use_cuda1 = False

    feature_file1 = "hand_crafted_features_global.csv"
    normalize_data1 = True

    # Initialize the simulator
    simulator1 = PatientDivisionSimulator(desired_classes=desired_classes1, n_rep=n_rep1,
                                          simulator_name=simulator_name1, working_dir=working_dir1,
                                          folder_name=folder_name1, data_group_dict=data_group_dict1,
                                          model_type=model_type1, train_perc=train_perc1, train_epochs=train_epochs1,
                                          train_lr=train_lr1, normalize_data=normalize_data1, use_cuda=use_cuda1,
                                          is_rehab=is_rehab1)
    '''simulator1 = PatientDivisionSimulator(desired_classes=desired_classes1, n_rep=n_rep1,
                                          simulator_name=simulator_name1, working_dir=working_dir1,
                                          folder_name=folder_name1, data_group_dict=data_group_dict1,
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
