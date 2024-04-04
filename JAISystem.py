# Import packages
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from SkeletonDataset import SkeletonDataset
from Trainer import Trainer
from NetworkTrainer import NetworkTrainer
from ExplainerType import ExplainerType


# Class
class JAISystem:
    models_fold = "results/models/models_for_JAI/"
    results_fold = "results/JAI/"

    def __init__(self, working_dir, system_name, model_name, use_keras=False):
        # Initialize attributes
        self.working_dir = working_dir
        self.models_dir = working_dir + self.models_fold
        self.results_dir = working_dir + self.results_fold

        if system_name not in os.listdir(self.results_dir):
            os.mkdir(self.results_dir + system_name)

        self.system_name = system_name
        self.model_name = model_name
        self.use_keras = use_keras

        self.trainer = Trainer.load_model(working_dir=working_dir, folder_name=None, model_name=model_name,
                                          use_keras=use_keras, folder_path=self.models_dir)
        self.trainer.show_model()

        self.explainer_type = None
        self.explainer = None

    def get_cam(self, item_name, target_layer, target_class, explainer_type, show=False):
        x, y = self.get_item_from_name(item_name)
        cam, output_prob = JAISystem.draw_cam(self.trainer, x, target_layer, target_class, explainer_type)

        title = item_name + " (" + str(int(y)) + ") - CAM " + str(target_class) + "(" + str(np.round(output_prob * 100,
                                                                                                     2)) + "%)"
        if not show:
            plt.figure()
            plt.imshow(cam, "jet")
            plt.title(title)
            plt.savefig(self.results_dir + self.system_name + "/" + item_name + "_" + explainer_type.value + target_layer +
                        "_" + str(target_class) + ".png", format="png", bbox_inches="tight", pad_inches=0)
            plt.close()
        else:
            self.show_skeleton(item=x, cam=cam, title=title)

    def show_skeleton(self, item, cam=None, title=None, max_color_range=63, slowing_parameter=3):
        # Extract x, y, z coordinates of joints
        if isinstance(item, str):
            positions, _ = self.get_item_from_name(item)
            title = item
        else:
            positions = item

        x = positions[:, 0::3]
        max_x = np.max(x)
        min_x = np.min(x)
        y = positions[:, 1::3]
        max_y = np.max(y)
        min_y = np.min(y)
        z = positions[:, 2::3]
        max_z = np.max(z)
        min_z = np.min(z)

        if cam.shape[1] == 1:
            is_1d = True
        else:
            is_1d = False

        if cam is not None and not is_1d:
            # Average CAM values among different axis of the same joint
            joint_cam = [np.mean(cam[:, (i * 3):(i + 1) * 3], axis=1) for i in range(cam.shape[1] // 3)]
            cam = np.stack(joint_cam, 1)

            # Adjust color range
            cam = JAISystem.normalize_cam(cam)
            cam = (max_color_range * cam).astype(int)

        # Plot each frame
        plt.ion()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")
        ax.set_aspect("auto")
        ax.view_init(elev=90, azim=-90)
        for frame in range(positions.shape[0]):
            # Initialize axis
            plt.cla()

            ax.axis("off")
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_zlim(min_z, max_z)
            ax.set_title(title)

            frame_x = x[frame]
            frame_y = y[frame]
            frame_z = z[frame]

            # Plot joints
            if cam is None or is_1d:
                color = "b"
                cmap = None
            else:
                color = cam[frame, :]
                cmap = "jet"
            ax.scatter(frame_x, frame_y, frame_z, c=color, cmap=cmap, marker="o", s=30, alpha=1)

            if cam is not None and is_1d:
                color = cam[frame]
                ax.scatter(max_x, max_y, max_z, c=color, cmap="jet", marker="o", s=500, alpha=1)

            # Plot connections
            for connection in SkeletonDataset.connections:
                joint1_pos = (frame_x[connection[0]], frame_y[connection[0]], frame_z[connection[0]])
                joint2_pos = (frame_x[connection[1]], frame_y[connection[1]], frame_z[connection[1]])
                ax.plot([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]], [joint1_pos[2], joint2_pos[2]],
                        c="r")

            plt.draw()
            plt.pause(SkeletonDataset.dt * slowing_parameter)
        plt.ioff()
        plt.show()

    def get_item_from_name(self, item_name):
        if item_name + SkeletonDataset.extension in self.trainer.train_data.data_files:
            dataset = self.trainer.train_data
        else:
            dataset = self.trainer.test_data

        x, y = dataset.get_item_from_name(item_name)
        return x, y

    @staticmethod
    def draw_cam(trainer, x, target_layer, target_class, explainer_type):
        # Adjust data
        net = trainer.net
        if isinstance(net.__dict__[target_layer], nn.Conv2d):
            is_2d = True
        elif isinstance(net.__dict__[target_layer], nn.Conv1d):
            is_2d = False
        else:
            is_2d = None
            print("The CAM method cannot be applied for the selected layer!")

        x = torch.from_numpy(x)
        x = x.unsqueeze(0)

        # Update activations and gradients
        net.eval()
        net.zero_grad()
        output, target_activation = net(x, layer_interrupt=target_layer)

        loss = output[:, target_class]
        loss.backward()
        target_grad = net.gradients

        # Compute CAM
        if explainer_type == ExplainerType.GC:
            cam = JAISystem.gc_map(target_activation, target_grad, is_2d)
        elif explainer_type == ExplainerType.HRC:
            cam = JAISystem.hrc_map(target_activation, target_grad)
        else:
            print("CAM generation method not implemented!")
            cam = None

        cam = cam.cpu().detach().numpy()
        cam = JAISystem.adjust_cam(cam, x, is_2d)

        output_prob = torch.softmax(output, dim=1)
        output_prob = output_prob[:, target_class]
        output_prob = output_prob.detach().numpy()[0]
        return cam, output_prob

    @staticmethod
    def gc_map(target_activation, target_grad, is_2d=True):
        target_activation = target_activation.squeeze(0)
        target_grad = target_grad.squeeze(0)

        if is_2d:
            dim_mean = (1, 2)
        else:
            dim_mean = 1
        weights = torch.mean(target_grad, dim=dim_mean)

        for i in range(target_activation.shape[0]):
            target_activation[i] *= weights[i]
        cam = torch.sum(target_activation, dim=0)
        cam = torch.relu(cam)

        return cam

    @staticmethod
    def hrc_map(target_activation, target_grad):
        target_activation = target_activation.squeeze(0)
        target_grad = target_grad.squeeze(0)

        for i in range(target_activation.shape[0]):
            target_activation[i] *= target_grad[i]
        cam = torch.sum(target_activation, dim=0)

        return cam

    @staticmethod
    def adjust_cam(map, x, is_2d=True):
        map = map.squeeze()
        map = JAISystem.normalize_cam(map)
        map = map.transpose()

        if is_2d:
            dim_reshape = (x.shape[2], x.shape[1])
        else:
            dim_reshape = (1, x.shape[1])
        map = cv2.resize(map, dim_reshape)
        map = np.uint8(255 * map)
        return map

    @staticmethod
    def normalize_cam(map):
        maximum = np.max(map)
        minimum = np.min(map)
        if maximum == minimum:
            if maximum == 1:
                map = np.ones(map.shape)
            else:
                map = np.zeros(map.shape)
        else:
            map = (map - minimum) / (maximum + minimum)

        return map


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"

    # Define the system
    model_name1 = "conv2d"
    system_name1 = "DD_" + model_name1
    use_keras1 = False
    system1 = JAISystem(working_dir=working_dir1, model_name=model_name1, system_name=system_name1,
                        use_keras=use_keras1)

    # Explain one item
    item_name1 = "S001C002P001R002A009"
    target_layer1 = "conv2"
    target_class1 = 0
    explainer_type1 = ExplainerType.GC
    show1 = True
    system1.get_cam(item_name=item_name1, target_layer=target_layer1, target_class=target_class1,
                    explainer_type=explainer_type1, show=show1)
