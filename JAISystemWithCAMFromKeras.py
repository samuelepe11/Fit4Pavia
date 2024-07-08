# Import packages
import numpy as np
import keras
import tensorflow as tf

from JAISystem import JAISystem
from JAISystemWithCAM import JAISystemWithCAM
from SkeletonDataset import SkeletonDataset
from ExplainerType import ExplainerType
from NetworkTrainer import NetworkTrainer


# Class
class JAISystemWithCAMFromKeras(JAISystemWithCAM):

    def __init__(self, working_dir, system_name, model_name):
        super().__init__(working_dir, system_name, model_name, use_keras=True)
        self.use_keras = True

    def get_cam(self, item_name, target_layer, target_class, explainer_type, show=False, switch_map_format=False,
                static_joints=False):
        x, y = self.get_item_from_name(item_name)
        cam, output_prob, bar_range = JAISystemWithCAMFromKeras.draw_cam(self.trainer, x, target_layer,
                                                                         target_class, explainer_type)
        x, dim = SkeletonDataset.remove_padding(x)
        x = x[0]
        dim = dim[0]
        cam = cam[:dim, :]
        self.display_output(item_name, target_layer, target_class, x, y, explainer_type, cam, output_prob,
                            switch_map_format, static_joints, show, bar_range)

    def get_item_from_name(self, item_name):
        item_name += SkeletonDataset.extension
        if item_name in self.trainer.descr_train:
            ind = self.trainer.descr_train.index(item_name)
            x, y = self.trainer.train_data
        else:
            ind = self.trainer.descr_test.index(item_name)
            x, y = self.trainer.test_data

        x = x[ind, :, :]
        x = np.array([x])
        y = y[ind]
        return x, y

    @staticmethod
    def draw_cam(trainer, x, target_layer, target_class, explainer_type, avoid_eval=False):
        net = trainer.net
        net.model.layers[-1].activation = None

        desired_layer = net.model.get_layer(target_layer)
        if isinstance(desired_layer, keras.layers.Conv2D):
            is_2d = True
        elif isinstance(desired_layer, keras.layers.Conv1D):
            is_2d = False
        else:
            is_2d = None
            print("The CAM method cannot be applied for the selected layer!")

        if trainer.normalize_input:
            x = (x - trainer.attr_mean) / trainer.attr_std
        x = tf.convert_to_tensor(x)

        # Extract activations and gradients
        grad_model = keras.models.Model([net.model.inputs], [desired_layer.output, net.model.output])
        with tf.GradientTape() as tape:
            target_activation, output = grad_model(x)
            target_score = output[:, target_class]

        if explainer_type != ExplainerType.VC:
            target_grad = tape.gradient(target_score, target_activation)
            target_grad = tf.squeeze(target_grad, axis=0)
        else:
            target_grad = None
        target_activation = tf.squeeze(target_activation, axis=0)

        # Compute CAM
        if explainer_type == ExplainerType.GC:
            cam = JAISystemWithCAMFromKeras.gc_map(target_activation, target_grad, is_2d)
        elif explainer_type == ExplainerType.HRC:
            cam = JAISystemWithCAMFromKeras.hrc_map(target_activation, target_grad, is_2d)
        elif explainer_type == ExplainerType.VC:
            # Check for the applicability of CAM algorithm
            last_layer = net.model.layers[-1]
            second_last_layer = net.model.layers[-2]
            if not isinstance(last_layer, keras.layers.Dense):
                print("The last layer should be a Fully Connected layer!")
            if not isinstance(second_last_layer, keras.layers.GlobalAveragePooling2D) and not isinstance(
                    second_last_layer, keras.layers.GlobalAveragePooling1D):
                print("The second last layer should be Global Average Pooling layer!")
            fc_weights = last_layer.weights[0]
            fc_weights = fc_weights[:, target_class]
            if len(fc_weights) != target_activation.shape[-1]:
                print("It is necessary to select the last convolutional layer!")
            cam = JAISystemWithCAMFromKeras.vc_map(target_activation, fc_weights, is_2d)
        else:
            print("Method not defined")
            cam = None
        cam = cam.numpy()
        cam, bar_range = JAISystem.adjust_map(cam, x, is_2d)

        output_prob = keras.activations.softmax(output)
        output_prob = output_prob[:, target_class]
        output_prob = output_prob.numpy()[0]
        net.model.layers[-1].activtion = "softmax"
        return cam, output_prob, bar_range

    @staticmethod
    def gc_map(target_activation, target_grad, is_2d=True):
        if is_2d:
            dim_mean = (0, 1)
            dim_channel = 2
        else:
            dim_mean = 0
            dim_channel = 1
        weights = tf.reduce_mean(target_grad, axis=dim_mean)

        feat_maps = []
        for i in range(target_activation.shape[-1]):
            if is_2d:
                activation = target_activation[:, :, i]
            else:
                activation = target_activation[:, i]
            feat_maps.append(activation * weights[i])
        target_activation = tf.stack(feat_maps, axis=dim_channel)
        cam = tf.reduce_sum(target_activation, axis=dim_channel)

        cam = tf.nn.relu(cam)
        return cam

    @staticmethod
    def hrc_map(target_activation, target_grad, is_2d=True):
        if is_2d:
            dim_channel = 2
        else:
            dim_channel = 1
        feat_maps = []
        for i in range(target_activation.shape[-1]):
            if is_2d:
                activation = target_activation[:, :, i]
                grad = target_grad[:, :, i]
            else:
                activation = target_activation[:, i]
                grad = target_grad[:, i]
            feat_maps.append(activation * grad)
        target_activation = tf.stack(feat_maps, axis=dim_channel)
        cam = tf.reduce_sum(target_activation, axis=dim_channel)

        cam = tf.nn.relu(cam)
        return cam

    @staticmethod
    def vc_map(target_activation, weights, is_2d=True):
        if is_2d:
            dim_channel = 2
        else:
            dim_channel = 1

        feat_maps = []
        for i in range(target_activation.shape[-1]):
            if is_2d:
                activation = target_activation[:, :, i]
            else:
                activation = target_activation[:, i]
            feat_maps.append(activation * weights[i])
        target_activation = tf.stack(feat_maps, axis=dim_channel)
        cam = tf.reduce_sum(target_activation, axis=dim_channel)

        cam = tf.nn.relu(cam)
        return cam


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    model_name1 = "conv1d_no_hybrid"
    system_name1 = "DD_" + model_name1
    system1 = JAISystemWithCAMFromKeras(working_dir=working_dir1, model_name=model_name1, system_name=system_name1)

    # Explain one item
    item_names = ["S002C002P012R002A008", "S013C002P018R002A008"]#, "S013C002P025R002A042", "S027C002P081R002A070",
                  #"S030C002P044R002A099"]
    target_layer1 = "conv1d_1"
    target_classes = range(2)
    #target_classes = range(15)
    explainer_types = [ExplainerType.GC, ExplainerType.HRC, ExplainerType.VC]
    show1 = False
    switch_map_format1 = False
    static_joints1 = False
    for item_name1 in item_names:
        for target_class1 in target_classes:
            for explainer_type1 in explainer_types:
                system1.get_cam(item_name=item_name1, target_layer=target_layer1, target_class=target_class1,
                                explainer_type=explainer_type1, show=show1, switch_map_format=switch_map_format1,
                                static_joints=static_joints1)
