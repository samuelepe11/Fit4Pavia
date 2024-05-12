# Import packages
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from SkeletonDataset import SkeletonDataset
from Trainer import Trainer


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
                                          use_keras=self.use_keras, folder_path=self.models_dir)
        self.trainer.show_model()

    def show_skeleton(self, item, map=None, title=None, switch_map_format=False, static_joints=False,
                      slowing_parameter=3):
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

        is_1d = False
        to_2d = False
        if map.shape[1] == 1:
            if switch_map_format:
                to_2d = True
            else:
                is_1d = True
                map = map.squeeze(1)
        else:
            if switch_map_format:
                is_1d = True
                map = np.mean(map, axis=1)
                map = JAISystem.normalize_map(map)
            else:
                if static_joints:
                    map = np.mean(map, axis=0)
                    map = JAISystem.normalize_map(map)

        if map is not None and not is_1d and not to_2d:
            if not static_joints:
                # Average map values among different axis of the same joint
                joint_map = [np.mean(map[:, (i * 3):(i + 1) * 3], axis=1) for i in range(map.shape[1] // 3)]
                map = np.stack(joint_map, 1)
            else:
                # Average map values among different axis of the same joint
                map = [np.mean(map[(i * 3):(i + 1) * 3]) for i in range(map.shape[0] // 3)]

            # Adjust color range
            map = JAISystem.normalize_map(map)

        # Plot each frame
        matplotlib.use("TkAgg")
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
            if map is None or is_1d:
                color = "gray"
                cmap = None
            else:
                if not static_joints:
                    color = map[frame]
                else:
                    color = map
                cmap = "jet"

            if not to_2d:
                # Draw joints with different colors
                ax.scatter(frame_x, frame_y, frame_z, c=color, cmap=cmap, marker="o", s=30, alpha=1)

            if map is not None and is_1d:
                color = plt.cm.jet(map[frame])
                ax.plot(max_x, max_y, max_z, color=color, marker="o", markersize=30)

            # Plot connections
            for connection in SkeletonDataset.connections:
                joint1_pos = (frame_x[connection[0]], frame_y[connection[0]], frame_z[connection[0]])
                joint2_pos = (frame_x[connection[1]], frame_y[connection[1]], frame_z[connection[1]])
                ax.plot([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]], [joint1_pos[2], joint2_pos[2]],
                        c="gray")

            if to_2d:
                # Draw joints of the same color
                color = plt.cm.jet(color)
                ax.plot(frame_x, frame_y, frame_z, color=color, marker="o", markersize=6, linewidth=0, alpha=1)

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

    def display_output(self, item_name, target_layer, target_class, x, y, explainer_type, map, output_prob,
                       switch_map_format=False, static_joints=False, show=False):
        title = "CAM for class " + str(target_class) + " (" + str(np.round(output_prob * 100, 2)) + "%) - true label: "\
                + str(int(y))
        if not show:
            plt.figure()
            plt.imshow(map, "jet")
            plt.title(title)
            plt.savefig(
                self.results_dir + self.system_name + "/" + item_name + "_" + explainer_type.value + target_layer +
                "_" + str(target_class) + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()
        else:
            self.show_skeleton(item=x, map=map, title=title, switch_map_format=switch_map_format,
                               static_joints=static_joints)

    @staticmethod
    def adjust_map(map, x, is_2d=True):
        map = JAISystem.normalize_map(map)

        if is_2d:
            dim_reshape = (x.shape[2], x.shape[1])
        else:
            dim_reshape = (1, x.shape[1])
        map = cv2.resize(map, dim_reshape)
        return map

    @staticmethod
    def normalize_map(map):
        maximum = np.max(map)
        minimum = np.min(map)
        if maximum == minimum:
            if maximum == 1:
                map = np.ones(map.shape)
            else:
                map = np.zeros(map.shape)
        else:
            map = (map - minimum) / (maximum + minimum)

        map = np.uint8(255 * map)
        return map
