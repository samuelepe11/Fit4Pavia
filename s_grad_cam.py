# Import packages
import random
import numpy as np
import torch
import keras
from signal_grad_cam import TorchCamBuilder, TfCamBuilder

from NetworkTrainer import NetworkTrainer
from SkeletonDataset import SkeletonDataset
from Trainer import Trainer

# Main
if __name__ == "__main__":
    # Define seeds
    seed = 111099
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.deterministic = True
    keras.utils.set_random_seed(seed)

    # Define variables
    working_dir1 = "./../"
    desired_classes1 = [8, 9]  # NTU HAR binary
    desired_classes1 = [7, 8, 9, 27, 42, 43, 46, 47, 54, 59, 60, 69, 70, 80, 99]  # NTU HAR multiclass
    # desired_classes1 = [1, 2]  # IntelliRehabDS correctness

    # Define the data
    train_perc = 0.7
    train_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                  group_dict={"C": 2, "R": 2}, data_perc=train_perc, divide_pt=True)
    test_data1 = SkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                 data_names=train_data1.remaining_instances)
    '''train_data1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1, data_perc=train_perc,
                                       divide_pt=True, maximum_length=200)
    test_data1 = RehabSkeletonDataset(working_dir=working_dir1, desired_classes=desired_classes1,
                                      data_names=train_data1.remaining_instances)'''

    # Define the model
    folder_name1 = "models_for_JAI"
    model_name1 = "conv1d_15classes"
    show_cm1 = True
    assess_calibration1 = True
    is_rehab1 = False

    # Load trained model
    use_keras1 = False
    trainer1 = Trainer.load_model(working_dir=working_dir1, folder_name=folder_name1, model_name=model_name1,
                                  use_keras=use_keras1, is_rehab=is_rehab1)

    avoid_eval1 = True
    trainer1.summarize_performance(show_process=True, show_cm=show_cm1, assess_calibration=assess_calibration1,
                                   avoid_eval=avoid_eval1, is_rehab=is_rehab1, ext_test_data_list=None)

    # Apply SignalGrad-CAM
    def preprocess_fc(signal):
        if trainer1.normalize_input:
            signal = (signal - trainer1.attr_mean) / trainer1.attr_std
        if not use_keras1:
            signal = torch.from_numpy(signal).float()
        return signal

    class_names = [train_data1.actions[desired_class - 1] for desired_class in desired_classes1]
    max_len = 199  # 118
    if not use_keras1:
        net = trainer1.set_cuda(trainer1.net, is_cuda=False)
        net.set_avoid_eval(avoid_eval1)
        cam_builder = TorchCamBuilder(model=net, transform_fn=preprocess_fc, class_names=class_names,
                                      input_transposed=True, ignore_channel_dim=True, extend_search=True, use_gpu=False,
                                      time_axs=0)
    else:
        cam_builder = TfCamBuilder(model=trainer1.net.model, transform_fn=preprocess_fc, class_names=class_names,
                                   ignore_channel_dim=True, input_transposed=False, padding_dim=max_len, time_axs=0)

    item_names = ["S011C002P038R002A009", "S010C002P021R002A008"]
    item_names = ["S029C002P049R002A069", "S031C002P099R002A080"]
    # item_names = ["306_18_8_8_1_stand", "201_18_2_23_2_wheelchair"]
    for name in item_names:
        try:
            x, y = trainer1.test_data.get_item_from_name(name)
        except AttributeError:
            x, y = test_data1.get_item_from_name(name)
        except TypeError:
            x, y = trainer1.train_data.get_item_from_name(name)

        data_list = [x]
        data_labels = [int(y)]

        target_classes = list(range(len(desired_classes1)))
        explainer_types = ["Grad-CAM", "HiResCAM"]
        target_layer = "conv2"
        addon = "" if not is_rehab1 else "../IntelliRehabDS/"
        result_dir_path = train_data1.working_dir + addon + "results/JAI_signal_grad_cam/" + model_name1 + "/"
        cams, probs, ranges = cam_builder.get_cam(data_list=data_list, data_labels=data_labels,
                                                  target_classes=target_classes, explainer_types=explainer_types,
                                                  target_layers=target_layer, softmax_final=True, data_names=[name],
                                                  data_sampling_freq=train_data1.fc, dt=0.5,
                                                  channel_names=train_data1.joint_names,
                                                  results_dir_path=result_dir_path, aspect_factor=10)
