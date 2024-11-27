# Import packages
import numpy as np
import torch
import torch.nn as nn
import cv2
import shap
import time
import lime_timeseries.lime_timeseries as lime_timeseries
import WindowSHAP.windowshap as windowshap
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image


from JAISystem import JAISystem
from SkeletonDataset import SkeletonDataset
from ExplainerType import ExplainerType
from NetworkTrainer import NetworkTrainer
from SetType import SetType


# Class
class JAISystemWithCAM(JAISystem):

    def __init__(self, working_dir, system_name, model_name, use_keras=False, avoid_eval=False):
        super().__init__(working_dir, system_name, model_name, use_keras, avoid_eval)

    def get_cam(self, item_name, target_layer, target_class, explainer_type, show=False, switch_map_format=False,
                static_joints=False, show_graphs=False, x=None, y=None):
        if item_name is not None:
            x, y = self.get_item_from_name(item_name)
        cam, output_prob, bar_range = JAISystemWithCAM.draw_cam(self.trainer, x, target_layer, target_class, explainer_type,
                                                                avoid_eval=self.avoid_eval)
        if item_name is not None:
            self.display_output(item_name, target_layer, target_class, x, y, explainer_type, cam, output_prob,
                                switch_map_format, static_joints, show, bar_range, show_graphs)
        else:
            return cam

    def get_item_from_name(self, item_name):
        if item_name + SkeletonDataset.extension in self.trainer.train_data.data_files:
            dataset = self.trainer.train_data
        else:
            dataset = self.trainer.test_data

        x, y = dataset.get_item_from_name(item_name)
        return x, y

    @staticmethod
    def draw_cam(trainer, x, target_layer, target_class, explainer_type, avoid_eval=False):
        net = trainer.net
        if isinstance(net.__dict__[target_layer], nn.Conv2d):
            is_2d = True
        elif isinstance(net.__dict__[target_layer], nn.Conv1d):
            is_2d = False
        else:
            is_2d = None
            print("The CAM method cannot be applied for the selected layer!")

        # Extract activations and gradients
        if avoid_eval:
            net.eval()
        else:
            net.set_training(False)
        net.zero_grad()

        if trainer.normalize_input:
            x = (x - trainer.attr_mean) / trainer.attr_std

        is_lime = explainer_type == ExplainerType.LIME
        is_shap = explainer_type == ExplainerType.SHAP
        is_gc_ref = explainer_type == ExplainerType.GCref or explainer_type == ExplainerType.HRCref
        is_ref = is_lime or is_shap or is_gc_ref
        if is_lime:
            cam = JAISystemWithCAM.lime_map(x, target_class, trainer, avoid_eval, is_2d)
        if is_shap:
            cam = JAISystemWithCAM.shap_map(x, target_class, trainer, avoid_eval, is_2d)

        x = torch.from_numpy(x)
        x = x.unsqueeze(0)

        if is_gc_ref:
            layer = net.__dict__[target_layer]
            net.set_avoid_eval(avoid_eval)
            if explainer_type == ExplainerType.GCref:
                method = GradCAM(model=net, target_layers=[layer])
            elif explainer_type == ExplainerType.HRCref:
                method = HiResCAM(model=net, target_layers=[layer])
            cam = method(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])
            cam = cam[0]
            cam = np.transpose(cam)
            cam = cv2.resize(cam, cam.shape)

        output, target_activation = net(x, layer_interrupt=target_layer, avoid_eval=avoid_eval)
        target_score = output[:, target_class]
        if not is_ref:
            target_score.backward()
            target_grad = net.gradients

            # Compute CAM
            target_activation = target_activation.squeeze(0)
            target_grad = target_grad.squeeze(0)
            if explainer_type == ExplainerType.GC:
                cam = JAISystemWithCAM.gc_map(target_activation, target_grad, is_2d)
            elif explainer_type == ExplainerType.HRC:
                cam = JAISystemWithCAM.hrc_map(target_activation, target_grad)

            cam = cam.cpu().detach().numpy()
            cam = cam.transpose()
        cam, bar_range = JAISystem.adjust_map(cam, x, is_2d, is_ref)

        output_prob = torch.softmax(output, dim=1)
        output_prob = output_prob[:, target_class]
        output_prob = output_prob.detach().numpy()[0]

        return cam, output_prob, bar_range

    @staticmethod
    def gc_map(target_activation, target_grad, is_2d=True):
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
        for i in range(target_activation.shape[0]):
            target_activation[i] *= target_grad[i]
        cam = torch.sum(target_activation, dim=0)

        cam = torch.relu(cam)
        return cam

    @staticmethod
    def lime_map(x, target_class, trainer, avoid_eval=False, is_2d=True):
        if is_2d:
            explainer = lime_image.LimeImageExplainer(kernel_width=0.25)
            explanation = explainer.explain_instance(x, lambda img: JAISystemWithCAM.pred_fc(img, trainer, avoid_eval, True),
                                                     labels=[target_class], top_labels=None, batch_size=32, num_features=10,
                                                     num_samples=500)
            x_shape = None
        else:
            explainer = lime_timeseries.LimeTimeSeriesExplainer(kernel_width=0.25)
            explanation = explainer.explain_instance(x, lambda sig: JAISystemWithCAM.pred_fc(sig, trainer, avoid_eval, True),
                                                     labels=[target_class], top_labels=None, num_features=10,
                                                     num_samples=500, num_slices=3, replacement_method="noise")
            x_shape = x.shape

        cam = JAISystem.adjust_lime_map(explanation, target_class, x_shape=x_shape)
        return cam

    @staticmethod
    def shap_map(x, target_class, trainer, avoid_eval=False, is_2d=True):
        if is_2d:
            masker = shap.maskers.Image("blur(128,128)", list(x.shape) + [1])
            explainer = shap.Explainer(lambda img: JAISystemWithCAM.pred_fc(img, trainer, avoid_eval, False),
                                       masker=masker, output_names=list(range(trainer.num_classes)))
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            shap_values = explainer(x, max_evals=5000, batch_size=32, outputs=[target_class])
            cam = shap_values.values[0, :, :, 0]
        else:
            background = []
            for _ in range(1):
                perturbation = np.random.normal(0, 3, x.shape)
                perturbed_sample = x + perturbation
                background.append(perturbed_sample)
            background = np.array(background)
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            x = np.array(x)
            explainer = windowshap.StationaryWindowSHAP(lambda img: JAISystemWithCAM.pred_fc(img, trainer, avoid_eval, False),
                                                        window_len=10, B_ts=background, test_ts=x, model_type="lstm")
            shap_values = explainer.shap_values(len(trainer.classes))
            cam = shap_values[target_class, 0, :, :]
            cam = np.sum(cam, axis=1, keepdims=True)
        cam = np.maximum(cam, 0)
        return cam

    @staticmethod
    def pred_fc(x, trainer, avoid_eval=False, is_lime=True):
        net = trainer.net
        x = torch.from_numpy(x)

        if avoid_eval:
            net.eval()
        else:
            net.set_training(False)

        with torch.no_grad():
            prediction = net(x, avoid_eval=avoid_eval, is_lime=is_lime)

        if is_lime:
            out = prediction.squeeze().numpy()
        else:
            out = prediction
        return out


# Main
if __name__ == "__main__":
    # Define seeds
    seed = 111099
    torch.manual_seed(seed)
    torch.backends.cuda.deterministic = True

    # Define variables
    working_dir1 = "./../"
    model_name1 = "conv1d"
    system_name1 = "DD_" + model_name1
    avoid_eval1 = True
    system1 = JAISystemWithCAM(working_dir=working_dir1, model_name=model_name1, system_name=system_name1,
                               avoid_eval=avoid_eval1)

    # Explain one item
    item_names = ["S011C002P038R002A009", "S010C002P021R002A008"]
    # item_names = ["S029C002P049R002A069", "S031C002P099R002A080"]
    target_layer1 = "conv2"
    target_classes = range(2)
    #target_classes = range(15)
    explainer_types = [ExplainerType.GC]
    show1 = True
    switch_map_format1 = False
    static_joints1 = False
    show_graphs1 = False
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
    set_type1 = SetType.TRAINING
    explainer_type1 = ExplainerType.GC
    # system1.average_explanations(set_type=set_type1, explainer_type=explainer_type1, target_layer=target_layer1)
