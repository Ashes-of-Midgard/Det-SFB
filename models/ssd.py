from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.detection.ssd import SSD, _vgg_extractor
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.vgg import vgg16
from torchvision.models.detection import _utils as det_utils


class SSD_Custom(SSD):
    def __init__(self,
                 backbone,
                 anchor_generator,
                 size, num_classes,
                 image_mean = None,
                 image_std = None,
                 head = None,
                 score_thresh = 0.01,
                 nms_thresh = 0.45,
                 detections_per_img = 200,
                 iou_thresh = 0.5,
                 topk_candidates = 400,
                 positive_fraction = 0.25,
                 **kwargs):
        super().__init__(backbone,
                         anchor_generator,
                         size,
                         num_classes,
                         image_mean,
                         image_std,
                         head,
                         score_thresh,
                         nms_thresh,
                         detections_per_img,
                         iou_thresh,
                         topk_candidates,
                         positive_fraction,
                         **kwargs)
        
    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if targets is not None:
            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                        )
                    )
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
            return detections, losses
        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

            return detections
        
    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)
        # bbox_regression: [batch_size, num_boxes, 4]
        # pred_scores: [batch_size, num_boxes, num_classes]

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []

        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            # boxes: [num_boxes, 4]
            # scores: [num_boxes, num_classes]
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections
    

def build_ssd_custom(num_classes:int) -> SSD_Custom:
    # Use custom backbones more appropriate for SSD
    backbone = vgg16(weights=None)
    backbone = _vgg_extractor(backbone, False, 5)
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],
    )

    kwargs = {
        # Rescale the input in a way compatible to the backbone
        "image_mean": [0.48235, 0.45882, 0.40784],
        "image_std": [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0],  # undo the 0-1 scaling of toTensor
    }
    model = SSD_Custom(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    return model