# Import packages
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from SkeletonDataset import SkeletonDataset
from Trainer import Trainer
from SetType import SetType


# Class
class JAISystem:
    models_fold = "results/models/models_for_JAI/"
    results_fold = "results/JAI/"

    def __init__(self, working_dir, system_name, model_name, use_keras=False, avoid_eval=False):
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

        self.avoid_eval = avoid_eval

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

    def show_graphs(self, item, map, folder, title=None):
        # Extract x, y, z coordinates of joints
        if isinstance(item, str):
            positions, _ = self.get_item_from_name(item)
            title = item
        else:
            positions = item

        if map.shape[1] == 1:
            map = np.tile(map, (1, positions.shape[1]))

        x = positions[:, 0::3]
        x_map = map[:, 0::3]
        y = positions[:, 1::3]
        y_map = map[:, 1::3]
        z = positions[:, 2::3]
        z_map = map[:, 2::3]

        # Draw single plots
        time_steps = np.arange(x.shape[0]) / SkeletonDataset.fc
        for j_set in SkeletonDataset.joint_set.keys():
            indices = SkeletonDataset.joint_set[j_set]
            plt.figure(figsize=(15, 10))
            plt.suptitle(title)
            for i in range(len(indices)):
                ind = indices[i]

                plt.subplot(len(indices), 3, 3 * i + 1)
                plt.tight_layout()
                plt.plot(time_steps, x[:, ind], color="black", linewidth=0.5)
                norm = mcolors.Normalize(vmin=0, vmax=0.1) if len(np.unique(x_map[:, ind])) == 1 and x_map[0][ind] == 0 else None
                plt.scatter(time_steps, x[:, ind], c=x_map[:, ind], cmap="jet", marker=".", s=40, norm=norm)
                plt.colorbar()
                plt.xlabel("Time (s)")
                plt.ylabel("X coordinate (m)")

                plt.subplot(len(indices), 3, 3 * i + 2)
                plt.plot(time_steps, y[:, ind], color="black", linewidth=0.5)
                norm = mcolors.Normalize(vmin=0, vmax=0.1) if len(np.unique(y_map[:, ind])) == 1 and y_map[0][ind] == 0 else None
                plt.scatter(time_steps, y[:, ind], c=y_map[:, ind], cmap="jet", marker=".", s=40, norm=norm)
                plt.colorbar()
                plt.xlabel("Time (s)")
                plt.ylabel("Y coordinate (m)")
                plt.title(SkeletonDataset.joint_names[ind].upper())

                plt.subplot(len(indices), 3, 3 * i + 3)
                plt.plot(time_steps, z[:, ind], color="black", linewidth=0.5)
                norm = mcolors.Normalize(vmin=0, vmax=0.1) if len(np.unique(z_map[:, ind])) == 1 and z_map[0][ind] == 0 else None
                plt.scatter(time_steps, z[:, ind], c=z_map[:, ind], cmap="jet", marker=".", s=40, norm=norm)
                plt.colorbar()
                plt.ylabel("Z coordinate (m)")
                plt.xlabel("Time (s)")
            plt.savefig(folder + "/" + j_set + ".png", format="png", bbox_inches="tight", pad_inches=0, dpi=500)
            plt.close()

    def get_item_from_name(self, item_name):
        if item_name + SkeletonDataset.extension in self.trainer.train_data.data_files:
            dataset = self.trainer.train_data
        else:
            dataset = self.trainer.test_data

        x, y = dataset.get_item_from_name(item_name)
        return x, y

    def display_output(self, item_name, target_layer, target_class, x, y, explainer_type, map, output_prob,
                       switch_map_format=False, static_joints=False, show=False, bar_range=None, show_graphs=False,
                       averaged_folder=None):
        if not self.use_keras:
            classes = self.trainer.train_data.classes
        else:
            classes = self.trainer.classes
        if averaged_folder is None:
            considered_class = SkeletonDataset.actions[classes[target_class] - 1]
            true_class = SkeletonDataset.actions[classes[int(y)] - 1]
            title = ("CAM for class '" + " ".join(considered_class.split(" ")[:2]) + "' (" + str(np.round(output_prob * 100, 2))
                     + "%) - true label: '" + " ".join(true_class.split(" ")[:2]) + "'")
        else:
            title = "Averaged CAM for class " + SkeletonDataset.actions[classes[target_class] - 1]
        if not show:
            if map.shape[1] > 1:
                plt.figure(figsize=(50, 50))
                aspect = "auto"
            else:
                plt.figure(figsize=(50, 2))
                aspect = 1

            norm = None
            if len(np.unique(map)) == 1:
                if np.unique(map) == 0:
                    norm = mcolors.Normalize(vmin=0, vmax=255)
                else:
                    map[0][0] = 0
            plt.matshow(np.transpose(map), aspect=aspect, cmap=plt.get_cmap("jet"), norm=norm)

            # Add colorbar
            cbar = plt.colorbar()
            if bar_range is not None:
                if bar_range[0] == 0:
                    minimum = int(bar_range[0])
                    min_str = str(minimum)
                else:
                    minimum = bar_range[0]
                    min_str = "{:.2e}".format(minimum)
                if bar_range[1] == minimum:
                    max_str = ""
                else:
                    max_str = "{:.2e}".format(bar_range[1])
                cbar.ax.get_yaxis().set_ticks([cbar.vmin, cbar.vmax], labels=[min_str, max_str])

            # Adjust axes
            plt.title(title)
            plt.xlabel("Time (s)")
            if averaged_folder is None:
                data_points = list(range(0, map.shape[0], 15))
                plt.xticks(data_points, [str(t / SkeletonDataset.fc) for t in data_points], fontsize=8)
            else:
                plt.gca().set_xticklabels([])
            if map.shape[1] > 1:
                plt.ylabel("Skeleton joint (3D coordinates)")
                plt.yticks(range(1, map.shape[1] + 1, 3), [s.upper() for s in SkeletonDataset.joint_names],
                           rotation=0, fontsize=8)
                for i in range(1, map.shape[1] // 3):
                    plt.axhline(i * 3 - 0.5, color="black", linestyle="--", linewidth=0.5)
            else:
                plt.yticks([], [])

            if averaged_folder is not None:
                item_name = averaged_folder
            else:
                if item_name not in os.listdir(self.results_dir + self.system_name):
                    os.mkdir(self.results_dir + self.system_name + "/" + item_name)
            if averaged_folder is None:
                addon = "/"
            else:
                addon = "_"
            path = (self.results_dir + self.system_name + "/" + item_name + addon + explainer_type.value + target_layer
                    + "_" + str(target_class))
            plt.savefig(path + ".png", format="png", bbox_inches="tight",
                        pad_inches=0, dpi=300)
            plt.close()

            if show_graphs and averaged_folder is None:
                if (explainer_type.value + target_layer + "_" + str(target_class) not in
                        os.listdir(self.results_dir + self.system_name + "/" + item_name)):
                    os.mkdir(path)
                self.show_graphs(item=x, map=map, title=title, folder=path)
        else:
            self.show_skeleton(item=x, map=map, title=title, switch_map_format=switch_map_format,
                               static_joints=static_joints)

    def average_explanations(self, set_type, explainer_type, target_layer):
        if set_type == SetType.TRAINING:
            dataset = self.trainer.train_data
        else:
            dataset = self.trainer.test_data

        # Get class averaged maps
        class_folder = set_type.value + "_class_averaged"
        if not self.use_keras:
            classes = self.trainer.train_data.classes
        else:
            classes = self.trainer.classes
        for label in range(len(classes)):
            class_list = []
            for i in range(len(dataset)):
                if not self.use_keras:
                    x, y = dataset.__getitem__(i)
                else:
                    x, y = dataset
                    x = x[i, :, :]
                    x = x[np.newaxis, :, :]
                    y = y[i]
                cams = self.get_cam(item_name=None, target_layer=target_layer, target_class=label,
                                    explainer_type=explainer_type, show=False, switch_map_format=False,
                                    static_joints=False, show_graphs=False, x=x, y=y)
                class_list.append(cams)

            dims = [c.shape[0] for c in class_list]
            class_list = [cv2.resize(np.array(c), (class_list[0].shape[1], int(np.mean(dims)))) for c in class_list]
            class_cam = np.mean(class_list, axis=0)
            self.display_output(item_name=None, target_layer=target_layer, target_class=label, x=None, y=None,
                                explainer_type=explainer_type, map=class_cam, output_prob=None, switch_map_format=False,
                                static_joints=False, show=False, bar_range=None, show_graphs=False,
                                averaged_folder=class_folder)

    @staticmethod
    def adjust_map(map, x, is_2d=True, is_lime=False):
        map, bar_range = JAISystem.normalize_map(map)
        if not is_lime:
            if is_2d:
                dim_reshape = (x.shape[2], x.shape[1])
            else:
                dim_reshape = (1, x.shape[1])
            map = cv2.resize(map, dim_reshape)

        return map, bar_range

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
            map = (map - minimum) / (maximum - minimum)

        map = np.uint8(255 * map)
        return map, (minimum, maximum)

    @staticmethod
    def adjust_lime_map(explanation, target_class, x_shape=None):
        scores = dict(explanation.local_exp[target_class])
        if x_shape is None:
            segm = explanation.segments
            cam = np.zeros_like(segm, np.float32)
            for i in range(segm.shape[0]):
                for j in range(segm.shape[1]):
                    segm_id = segm[i, j]
                    if segm_id in scores.keys():
                        cam[i, j] = scores[segm_id]
                    else:
                        cam[i, j] = scores[np.max(list(scores.keys()))]
        else:
            cam = np.zeros((x_shape[0], 1))
            for feature, weight in scores.items():
                time_step = feature // x_shape[1]
                cam[time_step] += weight

        cam = np.maximum(cam, 0)
        return cam

    @staticmethod
    def adjust_colormap(cmap):
        if len(np.unique(cmap)) == 1:
            if np.unique(cmap) == 0 or np.unique(cmap) == 1:
                cmap[0] = 0.5
        return cmap
