# Import packages
import torch
import torch.nn as nn

from JAISystem import JAISystem
from SkeletonDataset import SkeletonDataset
from ExplainerType import ExplainerType
from NetworkTrainer import NetworkTrainer


# Class
class JAISystemWithCAM(JAISystem):

    def __init__(self, working_dir, system_name, model_name, use_keras=False):
        super().__init__(working_dir, system_name, model_name, use_keras)

    def get_cam(self, item_name, target_layer, target_class, explainer_type, show=False, switch_map_format=False,
                static_joints=False):
        x, y = self.get_item_from_name(item_name)
        cam, output_prob = JAISystemWithCAM.draw_cam(self.trainer, x, target_layer, target_class, explainer_type)
        self.display_output(item_name, target_layer, target_class, x, y, explainer_type, cam, output_prob,
                            switch_map_format, static_joints, show)

    def get_item_from_name(self, item_name):
        if item_name + SkeletonDataset.extension in self.trainer.train_data.data_files:
            dataset = self.trainer.train_data
        else:
            dataset = self.trainer.test_data

        x, y = dataset.get_item_from_name(item_name)
        return x, y

    @staticmethod
    def draw_cam(trainer, x, target_layer, target_class, explainer_type):
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

        # Extract activations and gradients
        net.eval()
        net.zero_grad()
        output, target_activation = net(x, layer_interrupt=target_layer)
        target_score = output[:, target_class]
        target_score.backward()
        target_grad = net.gradients

        # Compute CAM
        target_activation = target_activation.squeeze(0)
        target_grad = target_grad.squeeze(0)
        if explainer_type == ExplainerType.GC:
            cam = JAISystemWithCAM.gc_map(target_activation, target_grad, is_2d)
        elif explainer_type == ExplainerType.HRC:
            cam = JAISystemWithCAM.hrc_map(target_activation, target_grad)
        else:
            print("CAM generation method not implemented!")
            cam = None
        cam = cam.cpu().detach().numpy()
        cam = cam.transpose()
        cam = JAISystem.adjust_map(cam, x, is_2d)

        output_prob = torch.softmax(output, dim=1)
        output_prob = output_prob[:, target_class]
        output_prob = output_prob.detach().numpy()[0]

        return cam, output_prob

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


# Main
if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../"

    # Define the system
    model_name1 = "conv2d"
    system_name1 = "DD_" + model_name1
    system1 = JAISystemWithCAM(working_dir=working_dir1, model_name=model_name1, system_name=system_name1)

    # Explain one item
    item_name1 = "S001C002P001R002A009"
    target_layer1 = "conv2"
    target_class1 = 1
    explainer_type1 = ExplainerType.GC
    show1 = True
    switch_map_format1 = False
    static_joints1 = True
    system1.get_cam(item_name=item_name1, target_layer=target_layer1, target_class=target_class1,
                    explainer_type=explainer_type1, show=show1, switch_map_format=switch_map_format1,
                    static_joints=static_joints1)
