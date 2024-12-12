# Import packages
import numpy as np
import random

from SkeletonDataset import SkeletonDataset


# Class
class RehabSkeletonDataset(SkeletonDataset):
    # Define class attributes
    data_fold = "../IntelliRehabDS/SkeletonData/Simplified/"
    results_fold = "../IntelliRehabDS/results/preliminary_analysis/"
    extension = ".txt"

    correctness = ["correct", "incorrect"]
    gestures = ["elbow flexion left", "elbow flexion right", "shoulder flexion left", "shoulder flexion right",
                "shoulder abduction left", "shoulder abduction right", "shoulder forward elevation", "side tap left",
                "side tap right"]
    actions = correctness + gestures

    list_pos = ["stand", "chair", "wheelchair", "Stand-frame"]
    n_positions = len(list_pos)

    list_pat = list(range(101, 108)) + list(range(201, 208)) + list(range(209, 218)) + list(range(301, 308))

    def __init__(self, working_dir, desired_classes, data_perc=None, divide_pt=False, data_names=None,
                 dataset_name=None, subfolder=None, maximum_length=None):
        super().__init__(working_dir, desired_classes, None, data_perc, divide_pt, data_names, dataset_name,
                         subfolder, is_rehab_data=True)
        self.n_patients = len(self.list_pat)

        # Remove files
        self.data_files = [x for x in self.data_files if x.endswith(self.extension) and x.split("_")[4] != "3"]

        # Define task
        self.is_correctness_label = np.all([c < 3 for c in desired_classes])

        # Read data
        for s in self.data_files:
            self.data.append(np.loadtxt(self.data_path + s, delimiter=","))

            if self.is_correctness_label:
                self.labels.append(int(s.split("_")[4]) - 1)
            else:
                self.labels.append(int(s.split("_")[2]))

        # Remove recordings longer than the maximum length
        self.maximum_length = maximum_length
        if maximum_length is not None:
            ind_longer = [i for i in range(len(self.data_files)) if self.data[i].shape[0] > maximum_length]
            ind_longer.sort(reverse=True)
            for i in ind_longer:
                self.data_files.pop(i)
                self.data.pop(i)
                self.labels.pop(i)

        # Separate train and test data
        if data_perc is not None:
            # Divide per patients
            if divide_pt:
                num_train_pt = round(self.n_patients * data_perc)
                train_pt = random.sample(range(self.n_patients), num_train_pt)
                train_pt = [self.list_pat[i] for i in train_pt]
                temp_data = []
                for pt in train_pt:
                    temp_data += self.find_elements(elements=self.data_files, group_dict={0: pt})

                self.remaining_instances = list(set(self.data_files) - set(temp_data))
                indices = [self.data_files.index(x) for x in temp_data]
                self.data_files = temp_data
                self.data = [self.data[i] for i in indices]
                self.labels = [self.labels[i] for i in indices]

            # Divide randomly
            else:
                self.data_dim = round(len(self.data_files) * data_perc)
                lists = list(zip(self.data_files, self.data, self.labels))
                random.shuffle(lists)
                self.data_files, self.data, self.labels = zip(*lists)

                self.remaining_instances = self.data_files[self.data_dim:]
                self.data_files = self.data_files[:self.data_dim]
                self.data = self.data[:self.data_dim]
                self.labels = self.labels[:self.data_dim]

        # Count data
        self.len = len(self.labels)

    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.labels[ind]

        return x.astype(np.float32), float(y)

    def __len__(self):
        return self.len

    @staticmethod
    def count_elements(elements, group_dict):
        count = len(RehabSkeletonDataset.find_elements(elements, group_dict))

        return count

    @staticmethod
    def find_elements(elements, group_dict):
        for k, v in group_dict.items():
            if k == 5 and v == "chair":
                elements = [x for x in elements if x.split(".")[0].split("_")[k] in [str(v), "sit"]]
            elif k == 2:
                elements = [x for x in elements if x.split(".")[0].split("_")[k] == str(v - 3)]
            else:
                elements = [x for x in elements if x.split(".")[0].split("_")[k] == str(v)]

        return elements

    @staticmethod
    def get_patient_ids(elements):
        all_pat = [int(x[:3]) for x in elements]
        unique_pat = set(all_pat)

        return unique_pat


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    desired_classes1 = [1, 2]
    desired_classes1 = list(range(3, 12))
    subfolder1 = "correctness_evaluation"
    subfolder1 = "gesture_evaluation"

    # Analyse data of the selected class
    dataset1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, subfolder=subfolder1)
    dataset1.show_statistics()
    dataset1.show_lengths()

    # Analyse data of the selected class with a fixed maximum length
    print("===========================================================================================================")
    dataset_name1 = "maxL200"
    maximum_length1 = 200

    dataset1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                    dataset_name=dataset_name1, subfolder=subfolder1, maximum_length=maximum_length1)
    dataset1.show_statistics()
    dataset1.show_lengths()
