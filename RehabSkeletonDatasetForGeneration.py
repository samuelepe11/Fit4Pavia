# Import packages
import numpy as np

from RehabSkeletonDataset import RehabSkeletonDataset


# Class
class RehabSkeletonDatasetForGeneration(RehabSkeletonDataset):
    # Define class attributes
    data_fold = "../IntelliRehabDS/SkeletonData/data_for_generation/"

    def __init__(self, working_dir, desired_classes, data_perc=None, divide_pt=False, data_names=None,
                 dataset_name=None, subfolder=None, maximum_length=None, waist_normalization=False,
                 extra_dir="real_files/"):
        super().__init__(working_dir, desired_classes, data_perc, divide_pt, data_names, dataset_name,
                         subfolder, maximum_length, extra_dir)
        if extra_dir != "real_files/":
            self.list_pat = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
                             418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,
                             436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453,
                             454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471,
                             472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489,
                             490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
                             508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525,
                             526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543,
                             544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561,
                             562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579,
                             580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597,
                             598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
                             616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633,
                             634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651,
                             652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669,
                             670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687,
                             688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705,
                             706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719]
            self.n_patients = len(self.list_pat)

        # Waist normalization
        if waist_normalization:
            self.waist_coords = [data[:, 18:21] for data in self.data]
            self.data = [self.data[i] - np.tile(self.waist_coords[i], (1, self.data[i].shape[1] // 3))
                         for i in range(len(self.data))]
            self.data = [np.delete(data, [18, 19, 20], axis=1) for data in self.data]


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    # desired_classes1 = [1, 2]
    # desired_classes1 = list(range(3, 12))
    desired_classes1 = [12, 13]
    filename1 = "syn_files_train"
    # subfolder1 = "correctness_evaluation_for_generation_" + filename1
    # subfolder1 = "gesture_evaluation_for_generation_" + filename1
    subfolder1 = "position_evaluation_for_generation_" + filename1

    # Analyse data of the selected classes
    dataset1 = RehabSkeletonDatasetForGeneration(working_dir=working_dir1, desired_classes=desired_classes1,
                                                 subfolder=subfolder1)
    # dataset1.show_statistics()
    # dataset1.show_lengths()

    # Analyse data of the selected classes with a fixed maximum length
    print("===========================================================================================================")
    dataset_name1 = ""
    maximum_length1 = None

    '''dataset1 = RehabSkeletonDatasetForGeneration(working_dir=working_dir1, desired_classes=desired_classes1,
                                                 dataset_name=dataset_name1, subfolder=subfolder1,
                                                 maximum_length=maximum_length1)'''
    # dataset1.show_statistics()
    # dataset1.show_lengths()

    # Analyse synthetic data of the selected classes with a fixed maximum length
    print("===========================================================================================================")
    dataset_name1 = "maxL200"
    maximum_length1 = 200

    dataset1 = RehabSkeletonDatasetForGeneration(working_dir=working_dir1, desired_classes=desired_classes1,
                                                 dataset_name=dataset_name1, subfolder=subfolder1,
                                                 maximum_length=maximum_length1,
                                                 extra_dir=filename1 + "/")
    dataset1.show_statistics()
    dataset1.show_lengths()
