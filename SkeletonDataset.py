# Import packages
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax

from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork


# Class
class SkeletonDataset(Dataset):
    # Define class attributes
    data_fold = "raw_npy/"
    results_fold = "results/preliminary_analysis/"
    extension = ".skeleton.npy"

    actions = ["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw",
               "sitting down", "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper",
               "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses",
               "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
               "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up",
               "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
               "pointing to something with finger", "taking a selfie", "check time (from watch)",
               "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together",
               "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)",
               "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)",
               "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm",
               "punching/slapping other person", "kicking other person", "pushing other person",
               "pat on back of other person", "point finger at the other person", "hugging other person",
               "giving something to other person", "touch other person's pocket", "handshaking",
               "walking towards each other", "walking apart from each other", "put on headphone", "take off headphone",
               "shoot at the basket", "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush (quite)",
               "flick hair", "thumb up", "thumb down", "make ok sign", "make victory sign",
               "staple book", "counting money", "cutting nails", "cutting paper (using scissors)", "snapping fingers",
               "open bottle", "sniff (smell)", "squat down", "toss a coin", "fold paper", "ball up paper",
               "play magic cube", "apply cream on face", "apply cream on hand back", "put on bag", "take off bag",
               "put something into a bag", "take something out of a bag", "open a box", "move heavy objects",
               "shake fist", "throw up cap/hat", "hands up (both hands)", "cross arms", "arm circles", "arm swings",
               "running on the spot", "butt kicks (kick backward)", "cross toe touch", "side kick", "yawn",
               "stretch oneself", "blow nose", "hit other person with something", "wield knife towards other person",
               "knock over other person (hit with body)", "grab other person's stuff",
               "shoot at other person with a gun", "step on foot", "high-five", "cheers and drink",
               "carry something with other person", "take a photo of other person", "follow other person",
               "whisper in other person's ear", "exchange things with other person", "support somebody with hand",
               "finger-guessing game (playing rock-paper-scissors)"]
    actions = [a.replace("/", " or ") for a in actions]

    n_patients = 40
    list_pat = list(range(1, n_patients + 1))
    n_cameras = 3
    list_cam = list(range(1, n_cameras + 1))
    n_setups = 32
    list_set = list(range(1, n_setups + 1))
    n_reps = 2
    list_rep = list(range(1, n_reps + 1))

    fc = 30
    dt = 1 / fc

    def __init__(self, working_dir, desired_classes, group_dict=None, data_perc=None, divide_pt=False, data_names=None,
                 dataset_name=None):
        self.working_dir = working_dir
        self.data_path = working_dir + self.data_fold
        self.results_dir = working_dir + self.results_fold
        self.classes = desired_classes

        if dataset_name is None:
            self.dataset_name = ""
        else:
            self.dataset_name = "_" + dataset_name

        # Find data
        self.data_files = os.listdir(self.data_path)

        # Initialise values
        self.remaining_instances = None
        self.other_class_instances = None

        if data_names is not None:
            self.data_files = data_names

        else:
            if desired_classes is not None:
                temp_data = []
                for c in desired_classes:
                    file_ending = f"{c:03}" + self.extension
                    temp_data += [x for x in self.data_files if x.endswith(file_ending)]
                self.data_files = temp_data

            # Select only the required data
            if group_dict is not None:
                self.data_files = SkeletonDataset.find_elements(elements=self.data_files, group_dict=group_dict)
                if "P" in group_dict.keys():
                    self.change_setting(group_dict["P"], "n_patients", "list_pat")
                if "C" in group_dict.keys():
                    self.change_setting(group_dict["C"], "n_cameras", "list_cam")
                if "S" in group_dict.keys():
                    self.change_setting(group_dict["S"], "n_setups", "list_set")
                if "R" in group_dict.keys():
                    self.change_setting(group_dict["R"], "n_reps", "list_rep")

            # Separate train and test data
            if data_perc is not None:
                # Divide per patients
                if divide_pt:
                    num_train_pt = round(self.n_patients * data_perc)
                    train_pt = random.sample(range(1, self.n_patients + 1), num_train_pt)
                    temp_data = []
                    for pt in train_pt:
                        temp_data += SkeletonDataset.find_elements(elements=self.data_files, group_dict={"P": pt})
                    self.remaining_instances = list(set(self.data_files) - set(temp_data))
                    self.data_files = temp_data
                # Divide randomly
                else:
                    self.data_dim = round(len(self.data_files) * data_perc)
                    random.shuffle(self.data_files)
                    self.remaining_instances = self.data_files[self.data_dim:]
                    self.data_files = self.data_files[:self.data_dim]

        # Read data
        self.data = []
        self.labels = []
        for s in self.data_files:
            self.data.append(np.load(self.data_path + s, allow_pickle=True).item())

            s = s.strip(self.extension)
            self.labels.append(int(s[-3:]))

        # Count data
        self.len = len(self.labels)
        self.seq_lens = None

    def __getitem__(self, ind):
        x = self.data[ind]["skel_body0"]
        y = self.classes.index(self.labels[ind])

        # RNN layer requires (sequence_length, features)
        x = x.reshape(x.shape[0], -1)

        return x.astype(np.float32), float(y)

    def __len__(self):
        return self.len

    def show_statistics(self):
        if self.dataset_name != "":
            print("STATISTICS FOR " + self.dataset_name.strip("_") + ":")
            print()

        # Print class distributions
        class_counts = []
        for c in self.classes:
            action = self.actions[c - 1]

            # Create folder to store results
            if action not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + action)

            # Number of elements belonging to the given class
            class_elements = SkeletonDataset.find_elements(elements=self.data_files, group_dict={"A": c})
            n_elements = len(class_elements)
            class_counts.append(n_elements)

            print("Items of class '" + action + "': " + str(n_elements))
            # Number of elements per class per patient
            pat_counts = []
            non_zero_pat = 0
            for pat in self.list_pat:
                pat_elements = SkeletonDataset.count_elements(elements=class_elements, group_dict={"P": pat})
                pat_counts.append(pat_elements)
                if pat_elements != 0:
                    print("> Items from patient " + str(pat) + ": " + str(pat_elements))
                    non_zero_pat += 1

            if c <= 60:
                trim = True
            else:
                trim = False
            self.create_bar_plot(counts=pat_counts, action=action, xlab="Patient", trim=trim)
            SkeletonDataset.print_separator()

            # Find the number of subjects that performed this action
            print(" > A total of " + str(non_zero_pat) + "/" + str(self.n_patients) + " subjects performed this action")

            # Find the number of subjects that performed this action and not the other ones
            exclusive_pat = self.find_exclusive_patients(ref_class=c)
            print(" > A total of " + str(len(exclusive_pat)) + "/" + str(self.n_patients) +
                  " subjects performed this action and not the other ones: " + str(list(exclusive_pat)))
            SkeletonDataset.print_separator()

            # Number of elements per class per camera
            cam_counts = []
            for cam in self.list_cam:
                cam_elements = SkeletonDataset.count_elements(elements=class_elements, group_dict={"C": cam})
                cam_counts.append(cam_elements)
                print("> Items from camera " + str(cam) + ": " + str(cam_elements))
            self.create_pie_plot(counts=cam_counts, action=action, label="Camera")
            SkeletonDataset.print_separator()

            # Number of elements per class per setup
            set_counts = []
            for setup in self.list_set:
                set_elements = SkeletonDataset.count_elements(elements=class_elements, group_dict={"S": setup})
                set_counts.append(set_elements)
                if set_elements != 0:
                    print("> Items from setup " + str(setup) + ": " + str(set_elements))
            self.create_bar_plot(counts=set_counts, action=action, xlab="Setup", trim=True)
            SkeletonDataset.print_separator()

            # Number of elements per class per repetition
            rep_counts = []
            for rep in self.list_rep:
                rep_elements = SkeletonDataset.count_elements(elements=class_elements, group_dict={"R": rep})
                rep_counts.append(rep_elements)
                print("> Items from replication " + str(rep) + ": " + str(rep_elements))
            self.create_pie_plot(counts=rep_counts, action=action, label="Repetition")
            SkeletonDataset.print_separator()

            # Number of elements per class per camera per repetition
            for cam in self.list_cam:
                for rep in self.list_rep:
                    cam_rep_elements = SkeletonDataset.count_elements(elements=class_elements,
                                                                      group_dict={"C": cam, "R": rep})
                    print("> Items from camera " + str(cam) + " and replication " + str(rep) + ": " +
                          str(cam_rep_elements))
            print("\n")

        # Draw class pie plot
        action_list = [self.actions[x - 1] for x in self.classes]
        if len(self.classes) <= 2:
            expand_dim = False
        else:
            expand_dim = True
        self.create_pie_plot(counts=class_counts, action=None, label=action_list, expand_dim=expand_dim)

    def find_exclusive_patients(self, ref_class):
        ref_elements = SkeletonDataset.find_elements(elements=self.data_files, group_dict={"A": ref_class})
        ref_set = SkeletonDataset.get_patient_ids(elements=ref_elements)

        for c in [x for x in self.classes if x != ref_class]:
            other_elements = SkeletonDataset.find_elements(elements=self.data_files, group_dict={"A": c})
            other_set = SkeletonDataset.get_patient_ids(elements=other_elements)
            ref_set -= other_set

        return ref_set

    def create_bar_plot(self, counts, action, xlab, trim=False):
        plt.figure(figsize=(15, 5))
        plt.bar(list(range(1, len(counts) + 1)), counts, width=0.9, color="seagreen")

        sns.despine()
        plt.title(action)
        plt.xlabel(xlab + " ID")
        plt.ylabel("Number of items in the dataset")

        # Set X values
        if trim:
            x_range = range(len(counts) + 2)
        else:
            x_range = np.arange(0, len(counts) + 2, step=5)
        plt.xticks(list(x_range))

        # Remove empty bars
        if trim:
            lims = []
            for i in range(len(counts)):
                if counts[i] != 0:
                    lims.append(i)
                    break
            for i in range(1, len(counts) + 1):
                if counts[len(counts) - i] != 0:
                    lims.append(len(counts) - i + 2)
                    break
            plt.xlim(lims)

        # Save the image
        img_path = self.results_dir + action + "/" + xlab + self.dataset_name + "_barplot.jpg"
        plt.savefig(img_path, dpi=300)
        plt.close()

    def create_pie_plot(self, counts, action, label, expand_dim=False):
        if not expand_dim:
            plt.figure(figsize=(4, 4))
        else:
            plt.figure(figsize=(10, 10))
        img_fold = self.results_dir
        if action is not None:
            img_fold += action + "/"

        if isinstance(label, list):
            if not expand_dim:
                labels = [x[:12] for x in label]
            else:
                labels = label
            if not expand_dim:
                action = labels[0] + " VS " + labels[1]
            else:
                action = "All Actions"
            label = action.replace(" ", "")
        else:
            labels = [label + " " + str(i + 1) for i in range(len(counts))]

        p, tx, text = plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140,
                              colors=sns.color_palette("Greens"))
        for i, a in enumerate(text):
            a.set_text("{}".format(counts[i]))
        plt.title(action)

        # Save the image
        img_path = img_fold + label + self.dataset_name + "_pieplot.jpg"
        plt.savefig(img_path, dpi=400)
        plt.close()

    def change_setting(self, group_dict_element, num_variable, list_variable):
        self.__dict__[num_variable] = np.array(group_dict_element).size
        if self.__dict__[num_variable] == 1:
            self.__dict__[list_variable] = [group_dict_element]
        else:
            self.__dict__[list_variable] = group_dict_element

    def show_lengths(self):
        lens = []
        for i in range(self.len):
            x, _ = self.__getitem__(i)
            lens.append(x.shape[0])
        self.seq_lens = lens
        lens = np.array(lens)

        minimum = np.min(lens)
        maximum = np.max(lens)
        avg = np.mean(lens)
        std = np.std(lens)
        print("Sequence length distribution... min = " + str(minimum) + ", max = " + str(maximum) + ", avg = " +
              str(np.round(avg)) + ", std = " + str(np.round(std, 2)))

        # Draw histogram
        plt.figure(figsize=(10, 5))
        plt.hist(lens, bins=100, color="seagreen")
        sns.despine()
        plt.ylabel("Absolute frequency")
        plt.xlabel("Sequence length")

        # Save the image
        img_path = self.results_dir + "/" + "sequence_length_histogram" + self.dataset_name + ".jpg"
        plt.savefig(img_path, dpi=300)
        plt.close()

    def get_list(self, to_tensor=True):
        data_list = []
        label_list = []
        for i in range(self.len):
            x, y = self.__getitem__(i)
            if to_tensor:
                x = torch.tensor(x)
            data_list.append(x)
            label_list.append(y)

        return data_list, label_list

    def min_max_scale(self):
        data_list, label_list = self.get_list(to_tensor=False)
        data_list = [TimeSeriesScalerMinMax().fit_transform([seq])[0] for seq in data_list]

        return data_list, label_list

    @staticmethod
    def count_elements(elements, group_dict):
        count = len(SkeletonDataset.find_elements(elements, group_dict))

        return count

    @staticmethod
    def find_elements(elements, group_dict):
        for k, v in group_dict.items():
            substr = k + f"{v:03}"
            elements = [x for x in elements if substr in x]

        return elements

    @staticmethod
    def get_patient_ids(elements):
        all_pat = [int(x[9:12]) for x in elements]
        unique_pat = set(all_pat)

        return unique_pat

    @staticmethod
    def print_separator():
        print("----------------------------------------------------------------------------------")

    @staticmethod
    def get_padded_datasets(train_data, test_data, train_dim):
        train_list, train_labels = train_data.get_list()
        test_list, test_labels = test_data.get_list()
        padded_seq = rnn_utils.pad_sequence(train_list + test_list, batch_first=True,
                                            padding_value=Conv1dNoHybridNetwork.mask_value)
        padded_seq = padded_seq.data.numpy()

        train_data = padded_seq[:train_dim, :, :]
        train_labels = np.asarray(train_labels)
        test_data = padded_seq[train_dim:, :, :]
        test_labels = np.asarray(test_labels)

        return (train_data, train_labels), (test_data, test_labels)


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    # desired_classes1 = [8, 9]
    desired_classes1 = list(range(1, 11))

    # Analyse data of the selected class
    dataset1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, group_dict={"C": 2, "R": 2},
                               dataset_name="C2R2")
    dataset1.show_statistics()
    dataset1.show_lengths()

    # Analyse data of the selected class
    dataset1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1)
    dataset1.show_statistics()
    dataset1.show_lengths()
