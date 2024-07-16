# Import packages
import torch
import torch.nn as nn
from lime import lime_image

from JAISystem import JAISystem
from SkeletonDataset import SkeletonDataset
from ExplainerType import ExplainerType
from NetworkTrainer import NetworkTrainer


# Class
class JAISystemWithCAM(JAISystem):

    def __init__(self, working_dir, system_name, model_name, use_keras=False, avoid_eval=False):
        super().__init__(working_dir, system_name, model_name, use_keras, avoid_eval)

    def get_cam(self, item_name, target_layer, target_class, explainer_type, show=False, switch_map_format=False,
                static_joints=False):
        x, y = self.get_item_from_name(item_name)
        cam, output_prob, bar_range = JAISystemWithCAM.draw_cam(self.trainer, x, target_layer, target_class, explainer_type,
                                                                avoid_eval=self.avoid_eval)
        self.display_output(item_name, target_layer, target_class, x, y, explainer_type, cam, output_prob,
                            switch_map_format, static_joints, show, bar_range)

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
        if is_lime:
            cam = JAISystemWithCAM.lime_map(x, target_class, trainer, avoid_eval)

        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        output, target_activation = net(x, layer_interrupt=target_layer, avoid_eval=avoid_eval)
        target_score = output[:, target_class]
        if not is_lime:
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
        cam, bar_range = JAISystem.adjust_map(cam, x, is_2d)

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
    def lime_map(x, target_class, trainer, avoid_eval=False):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(x, lambda img: JAISystemWithCAM.lime_pred_fc(img, trainer, avoid_eval),
                                                 labels=[target_class], top_labels=None)
        _, cam = explanation.get_image_and_mask(label=target_class, positive_only=True, hide_rest=True)
        return cam

    @staticmethod
    def lime_pred_fc(x, trainer, avoid_eval=False):
        net = trainer.net
        x = torch.from_numpy(x)

        if avoid_eval:
            net.eval()
        else:
            net.set_training(False)

        with torch.no_grad():
            prediction = net(x, avoid_eval=avoid_eval, is_lime=True)
        return prediction.squeeze().numpy()


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"
    model_name1 = "conv2d_15classes"
    system_name1 = "DD_" + model_name1
    avoid_eval1 = True
    system1 = JAISystemWithCAM(working_dir=working_dir1, model_name=model_name1, system_name=system_name1,
                               avoid_eval=avoid_eval1)

    # Explain one item
    item_names = (["S007C002P007R002A009", "S016C002P019R002A009", "S011C002P038R002A007", "S019C002P047R002A070",
                   "S027C002P081R002A070"])
    target_layer1 = "conv2"
    target_classes = range(2)
    target_classes = range(15)
    explainer_types = [ExplainerType.LIME]
    show1 = False
    switch_map_format1 = False
    static_joints1 = False
    for item_name1 in item_names:
        for target_class1 in target_classes:
            for explainer_type1 in explainer_types:
                system1.get_cam(item_name=item_name1, target_layer=target_layer1, target_class=target_class1,
                                explainer_type=explainer_type1, show=show1, switch_map_format=switch_map_format1,
                                static_joints=static_joints1)
