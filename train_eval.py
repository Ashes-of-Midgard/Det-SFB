from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

from models.det_sfb import DetSFB
from datasets.coco_detection import COCO_CAT_ID_CONVERT

COCO_CAT_ID_CONVERT_INVERSE = list(COCO_CAT_ID_CONVERT.keys()).sort()

def train_one_epoch(
        epoch: int, model: DetSFB, train_loader: DataLoader,
        optimizer_det: Optimizer, optimizer_sup: Optimizer,
        device:Union[str, torch.device], no_targets=False):
    model.train()
    for itr, samples in enumerate(train_loader):
        if not no_targets:
            images, targets = samples
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            _, loss_det, loss_sup = model.forward(images, targets)
            model.freeze_sup()
            loss_det_value = loss_det['loss_cls'] + loss_det['loss_box']
            loss_det_value.backward(retain_graph=True)
            optimizer_det.step()
            model.unfreeze_sup()

            model.freeze_det()
            loss_sup_value = loss_sup['loss_adv'] + loss_sup['loss_cls_sup']
            loss_sup_value.backward()
            optimizer_sup.step()
            model.unfreeze_det()

            loss_value = loss_det_value + loss_sup_value
            loss = loss_det.update(loss_sup)
        else:
            try:
                images = samples
            except:
                images, _ = samples
            images = list(image.to(device) for image in images)

            _, loss_det = model.forward(images)
            model.freeze_sup()
            loss_det_value = loss_det['loss_cls'] + loss_det['loss_box']
            loss_det_value.backward()
            optimizer_det.step()
            model.unfreeze_sup()

            loss_value = loss_det_value
            loss = loss_det

        if itr % 100 == 0:
            loss_string = ""
            for k, v in loss.items():
                loss_string += f"{k}: {v:.4f} "
            print(f"Epoch [{epoch}] Iteration #{itr} loss: {loss_value} {loss_string}")


def evaluate_model(model: DetSFB, eval_loader: DataLoader, coco_gt: COCO) -> float:
    model.eval()
    with torch.no_grad():
        all_predicted_results = []
        for images, targets in eval_loader:
            predictions = model.forward(images)

            for idx, prediction in enumerate(predictions):
                image_id = targets[idx]['image_id']
                predicted_boxes = np.array(prediction['boxes'].cpu()).astype(np.int32)
                predicted_scores = np.array(prediction['scores'].cpu())
                predicted_labels = np.array(prediction['labels'].argmax().cpu())

                for box, score, label in zip(predicted_boxes, predicted_scores, predicted_labels):
                    x_min, y_min, x_max, y_max = box
                    predicted_result = {
                        "image_id": image_id,
                        "category_id": COCO_CAT_ID_CONVERT_INVERSE[label.item()],
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "score": score
                    }
                    all_predicted_results.append(predicted_result)

    coco_dt = coco_gt.loadRes(all_predicted_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mean_mAP = coco_eval.stats[0]
    return mean_mAP