from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


COCO_CAT_ID_CONVERT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                       11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
                       20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
                       31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
                       39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
                       48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
                       56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
                       64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
                       76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
                       85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_transform():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ToGray(p=0.01),
        OneOf([
            # IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            CLAHE(),
            # IAASharpen(),
            # IAAEmboss(),
            RandomBrightnessContrast(),
            ], p=0.25),
        HueSaturationValue(p=0.25),
        Resize(224, 224),
        ToTensorV2()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return Compose([
        Resize(224, 224),
        ToTensorV2(),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


class TransformsCOCO(nn.Module):
    def __init__(self, train=True):
        super().__init__()
        if train:
            self.transforms = get_train_transform()
        else:
            self.transforms = get_valid_transform()
    
    def forward(self, image, target_list):
        image = np.array(image, np.float32)
        boxes = [target['bbox'] for target in target_list]
        labels = [COCO_CAT_ID_CONVERT[target['category_id']] for target in target_list]

        boxes = np.array(boxes)
        labels = np.array(labels)

        # convert [x,y,w,h] to [x_min,y_min,x_max,y_max]
        boxes[:,2:] = boxes[:,2:] + boxes[:,:2]
        
        target = dict()
        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
        image = transformed['image']
        target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)

        return image, target
    

def get_coco_dataset_and_loader(data_dir, batch_size, train) -> Tuple[Dataset, DataLoader]:
    if train:
        dataset = torchvision.datasets.CocoDetection(root=f'{data_dir}/train2017',
                                                     annFile=f'{data_dir}/annotations/instances_train2017.json',
                                                     transforms=TransformsCOCO(train))
        loader= DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, collate_fn=collate_fn)
    else:
        dataset = torchvision.datasets.CocoDetection(root=f'{data_dir}/val2017',
                                                     annFile=f'{data_dir}/annotations/instances_val2017.json',
                                                     transforms=TransformsCOCO(train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size, collate_fn=collate_fn)
    
    return dataset, loader