from typing import List, Tuple, Dict, Union
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.resnet import Bottleneck
from torchvision.transforms import functional as trans_F
from .ssd import build_ssd_custom


class DetSFB(nn.Module):
    """ Detector trained with Supervisor's FeedBacks
    """
    def __init__(self, det_net_dict, sup_net_dict, roi_size=(32, 32)):
        super().__init__()
        self.det_net_dict = det_net_dict
        self.sup_net_dict = sup_net_dict
        self.det_net = build_det_net(det_net_dict)
        self.sup_net = build_sup_net(sup_net_dict)
        self.roi_size = roi_size

    def forward(
            self, images:List[Tensor], targets:List[Dict]=None
    ) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Dict[str, Tensor]], Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]]:
        det_outputs = self.det_net(images)
        if self.training:
            rois = []
            roi_labels = []
            for image, output in zip(images, det_outputs):
                boxes = output['boxes']
                labels = output['labels']
                scores = output['scores']
                score_threshold = sorted(list(scores))[int(0.5*len(scores))]
                for box, label, score in zip(boxes, labels, scores):
                    if score > score_threshold:
                        rois.append(self.roi_align(image, box, size=self.roi_size))
                        roi_labels.append(label)
            rois = torch.stack(rois)
            roi_labels = torch.stack(roi_labels)
            sup_outputs = self.sup_net(rois)
            loss_det = self.loss_det(sup_outputs, roi_labels)
            
            if targets is not None:
                rois_gt = []
                roi_labels_gt = []
                for image, target in zip(images, targets):
                    boxes = target['boxes']
                    labels = target['labels']
                    for box, label in zip(boxes, labels):
                        box_converted = torch.zeros_like(box)
                        box_converted[:1] = box[:1]
                        box_converted[2:] = box[:1] + box[2:]
                        rois_gt.append(self.roi_align(image, box, size=self.roi_size))
                        roi_labels_gt.append(label)
                rois_gt = torch.stack(rois_gt)
                roi_labels_gt = torch.stack(roi_labels_gt)
                rois_synth = rois.detach()
                rois = torch.concat(rois_gt, rois_synth)
                sup_outputs = self.sup_net(rois)
                loss_sup = self.loss_sup(sup_outputs, roi_labels_gt)
                return det_outputs, loss_det, loss_sup
            
            return det_outputs, loss_det
        else:
            return det_outputs
        
    def roi_align(self, image: Tensor, box: Tensor, size: Union[List, Tuple]=None) -> Tensor:
        x_min, y_min, x_max, y_max = box
        height, width = image.shape[1], image.shape[2]

        x_min = torch.clamp(x_min, min=0, max=image.shape[1]-1)
        y_min = torch.clamp(y_min, min=0, max=image.shape[0]-1)
        x_max = torch.clamp(x_max, min=int(x_min)+1, max=image.shape[1])
        y_max = torch.clamp(y_max, min=int(y_min)+1, max=image.shape[0])

        # 将框坐标转换为 [-1, 1] 范围的归一化坐标
        grid_x_min = 2 * (x_min / width) - 1
        grid_y_min = 2 * (y_min / height) - 1
        grid_x_max = 2 * (x_max / width) - 1
        grid_y_max = 2 * (y_max / height) - 1

        # 创建一个 1x2x2x2 的采样网格，表示裁剪区域的坐标 (注意: `grid_sample` 需要 N x H x W x 2 格式)
        grid = torch.tensor([[[[grid_x_min, grid_y_min], [grid_x_max, grid_y_min]],
                            [[grid_x_min, grid_y_max], [grid_x_max, grid_y_max]]]],
                            dtype=torch.float32,
                            device=image.device)

        # 通过 grid_sample 函数进行裁剪
        cropped_image = F.grid_sample(image.unsqueeze(0), grid, align_corners=True)[0]
        if size is not None:
            cropped_image = trans_F.resize(cropped_image, size)
        return cropped_image

    def loss_det(self, sup_outputs: Tensor, roi_labels: Tensor) -> Tensor:
        sup_labels, sup_scores = sup_outputs
        loss_cls = F.cross_entropy(roi_labels, sup_labels)
        loss_box = -torch.mean(torch.log(sup_scores))
        return {'loss_cls': loss_cls, 'loss_box': loss_box}
    
    def loss_sup(self, sup_outputs: Tuple[Tensor, Tensor], roi_labels_gt: Tensor) -> Tensor:
        sup_labels, sup_scores = sup_outputs
        sup_labels = sup_labels[:len(roi_labels_gt)]
        sup_scores_pos =sup_scores[:len(roi_labels_gt)]
        sup_scores_neg = sup_scores[len(roi_labels_gt):]
        loss_adv = -torch.mean(torch.log(sup_scores_pos))-torch.mean(torch.log(1-sup_scores_neg))
        loss_cls = F.cross_entropy(sup_labels, roi_labels_gt)
        return {'loss_adv': loss_adv, 'loss_cls_sup': loss_cls}
    
    def freeze_det(self):
        for param in self.det_net.parameters():
            param.requires_grad_(False)

    def unfreeze_det(self):
        for param in self.det_net.parameters():
            param.requires_grad_(True)

    def freeze_sup(self):
        for param in self.sup_net.parameters():
            param.requires_grad_(False)

    def unfreeze_sup(self):
        for param in self.sup_net.parameters():
            param.requires_grad_(True)


def build_det_net(net_dict) -> nn.Module:
    num_classes = net_dict['num_classes']
    if net_dict['name'] == 'ssd':
        det_net = build_ssd_custom(num_classes)
    return det_net


def build_sup_net(net_dict) -> nn.Module:
    num_classes = net_dict['num_classes']
    if net_dict['name'] == 'resnet':
        resnet_backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
        sup_net = SupvisorNet(resnet_backbone, num_classes)
    return sup_net


class SupvisorNet(nn.Module):
    def __init__(self, backbone:nn.Module, num_classes:int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.out_channels, num_classes)
        self.evaluator = nn.Linear(backbone.out_channels, 1)

    def forward(self, x:torch.Tensor) -> Tuple[Tensor, Tensor]:
        features = self.backbone(x)
        labels = self.classifier(features)
        scores = self.evaluator(features)
        return labels, scores
    

class ResNetBackbone(models.ResNet):
    def __init__(self,
                 block,
                 layers,
                 num_classes = 1000,
                 zero_init_residual = False,
                 groups = 1,
                 width_per_group = 64,
                 replace_stride_with_dilation = None,
                 norm_layer = None):
        super().__init__(block,
                         layers,
                         num_classes,
                         zero_init_residual,
                         groups,
                         width_per_group,
                         replace_stride_with_dilation,
                         norm_layer)
        self.out_channels = self.fc.in_features
        del self.fc

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
