# Import packages
import numpy as np
import keras
import tensorflow as tf
import shap
import time
import lime_timeseries.lime_timeseries as lime_timeseries
import WindowSHAP.windowshap as windowshap
from lime import lime_image

from JAISystem import JAISystem
from JAISystemWithCAM import JAISystemWithCAM
from SkeletonDataset import SkeletonDataset
from ExplainerType import ExplainerType
from NetworkTrainer import NetworkTrainer
from Conv1dNoHybridNetwork import Conv1dNoHybridNetwork
from SetType import SetType


# Class
class JAISystemWithCAMFromKeras(JAISystemWithCAM):

    def __init__(self, working_dir, system_name, model_name):
        super().__init__(working_dir, system_name, model_name, use_keras=True)
        self.use_keras = True

    def get_cam(self, item_name, target_layer, target_class, explainer_type, show=False, switch_map_format=False,
                static_joints=False, show_graphs=False, x=None, y=None):
        if item_name is not None:
            x, y = self.get_item_from_name(item_name)
        cam, output_prob, bar_range = JAISystemWithCAMFromKeras.draw_cam(self.trainer, x, target_layer,
                                                                         target_class, explainer_type)
        x, dim = SkeletonDataset.remove_padding(x)
        x = x[0]
        dim = dim[0]
        cam = cam[:dim, :]
        if item_name is not None:
            self.display_output(item_name, target_layer, target_class, x, y, explainer_type, cam, output_prob,
                                switch_map_format, static_joints, show, bar_range, show_graphs)
        else:
            return cam

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

        is_lime = explainer_type == ExplainerType.LIME
        is_shap = explainer_type == ExplainerType.SHAP
        is_ref = is_lime or is_shap
        if is_lime:
            cam = JAISystemWithCAMFromKeras.lime_map(x, target_class, trainer, avoid_eval, is_2d)
        if is_shap:
            cam = JAISystemWithCAMFromKeras.shap_map(x, target_class, trainer, avoid_eval, is_2d)

        net.model.layers[-1].activation = None
        x = tf.convert_to_tensor(x)

        # Extract activations and gradients
        grad_model = keras.models.Model([net.model.inputs], [desired_layer.output, net.model.output])
        with tf.GradientTape() as tape:
            target_activation, output = grad_model(x)
            target_score = output[:, target_class]

        if explainer_type != ExplainerType.VC and not is_ref:
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

        if not is_ref:
            cam = cam.numpy()
        cam, bar_range = JAISystem.adjust_map(cam, x, is_2d, is_lime)

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

    @staticmethod
    def lime_map(x, target_class, trainer, avoid_eval=False, is_2d=True):
        _, dim = SkeletonDataset.remove_padding(x)
        x = x.squeeze(0)
        if is_2d:
            explainer = lime_image.LimeImageExplainer(kernel_width=0.25)
            explanation = explainer.explain_instance(x, lambda img: JAISystemWithCAMFromKeras.pred_fc(img, trainer, pad_dim=dim),
                                                     labels=[target_class], top_labels=None, batch_size=32, num_features=10,
                                                     num_samples=500)
            x_shape = None
        else:
            explainer = lime_timeseries.LimeTimeSeriesExplainer(kernel_width=0.25)
            explanation = explainer.explain_instance(x, lambda img: JAISystemWithCAMFromKeras.pred_fc(img, trainer, pad_dim=dim, is_2d=is_2d),
                                                     labels=[target_class], top_labels=None, num_features=10,
                                                     num_samples=500, num_slices=3, replacement_method="noise")
            x_shape = x.shape

        cam = JAISystem.adjust_lime_map(explanation, target_class, x_shape=x_shape)
        return cam

    @staticmethod
    def shap_map(x, target_class, trainer, avoid_eval=False, is_2d=True):
        _, dim = SkeletonDataset.remove_padding(x)
        if is_2d:
            masker = shap.maskers.Image("inpaint_telea", list(x.shape[1:]) + [1])
            explainer = shap.Explainer(lambda img: JAISystemWithCAMFromKeras.pred_fc(img, trainer, avoid_eval, False, pad_dim=dim),
                                       masker=masker, output_names=(list(range(trainer.num_classes))))
            shap_values = explainer(x, max_evals=5000, batch_size=32, outputs=[target_class])
            cam = shap_values.values[0, :, :, 0]
        else:
            background = []
            for _ in range(1):
                perturbation = np.random.normal(0, 3, x.shape)
                perturbed_sample = x + perturbation
                background.append(perturbed_sample[0])
            background = np.array(background)
            explainer = windowshap.StationaryWindowSHAP(lambda img: JAISystemWithCAMFromKeras.pred_fc(img, trainer, avoid_eval, False, pad_dim=dim),
                                                        window_len=10, B_ts=background, test_ts=x, model_type="lstm")
            shap_values = explainer.shap_values(len(trainer.classes))
            cam = shap_values[target_class, 0, :, :]
            cam = np.sum(cam, axis=1, keepdims=True)

        cam = np.maximum(cam, 0)
        return cam

    @staticmethod
    def pred_fc(x, trainer, avoid_eval=False, is_lime=True, pad_dim=None, is_2d=True):
        net = trainer.net
        if is_lime and is_2d:
            x = np.mean(x, axis=-1, keepdims=True)
        if pad_dim is not None:
            if is_lime and is_2d:
                x[:, pad_dim[0]:, :, :] = Conv1dNoHybridNetwork.mask_value
            else:
                x[:, pad_dim[0]:, :] = Conv1dNoHybridNetwork.mask_value
        x = tf.convert_to_tensor(x)

        prediction = net.predict(x)
        return prediction


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 111099
    keras.utils.set_random_seed(seed)

    # Define variables
    working_dir1 = "./../"
    model_name1 = "conv2d_no_hybrid_15classes"
    system_name1 = "DD_" + model_name1
    system1 = JAISystemWithCAMFromKeras(working_dir=working_dir1, model_name=model_name1, system_name=system_name1)

    # Explain one item
    # item_names = ["S011C002P038R002A009", "S010C002P021R002A008"]
    item_names = ["S029C002P049R002A069", "S031C002P099R002A080"]
    target_layer1 = "conv2d_97"
    # target_classes = range(2)
    target_classes = range(15)
    explainer_types = [ExplainerType.GC, ExplainerType.HRC, ExplainerType.VC, ExplainerType.LIME, ExplainerType.SHAP]
    show1 = False
    switch_map_format1 = False
    static_joints1 = False
    show_graphs1 = True
    print()
    for item_name1 in item_names:
        for explainer_type1 in explainer_types:
            print("Creating", explainer_type1.value, "maps for", item_name1 + "...")
            start = time.time()
            for target_class1 in target_classes:
                system1.get_cam(item_name=item_name1, target_layer=target_layer1, target_class=target_class1,
                                explainer_type=explainer_type1, show=show1, switch_map_format=switch_map_format1,
                                static_joints=static_joints1, show_graphs=show_graphs1)
            end = time.time()
            print(" > duration:", round((end - start) / 60, 4), "min")
            print()

    # Average explanations
    set_types = [SetType.TRAINING, SetType.TEST]
    explainer_type1 = ExplainerType.GC
    for set_type1 in set_types:
        system1.average_explanations(set_type=set_type1, explainer_type=explainer_type1, target_layer=target_layer1)
