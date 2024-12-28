import os

import torch
from torch.optim import SGD
import torchvision

from models.det_sfb import DetSFB
from datasets.coco_detection import get_coco_dataset_and_loader
from train_eval import train_one_epoch, evaluate_model


if __name__=="__main__":
    data_dir = 'D:\Data\Dataset\coco_2017'
    os.makedirs("./logs",exist_ok=True)


    train_dataset, train_loader = get_coco_dataset_and_loader(data_dir, 4, train=True)
    val_dataset, val_loader = get_coco_dataset_and_loader(data_dir, 4, train=False)
    val_dataset: torchvision.datasets.CocoDetection

    # 定义模型
    det_net_dict = {"name":"ssd", "num_classes": 80}
    sup_net_dict = {"name":"resnet", "num_classes":80}
    model = DetSFB(det_net_dict, sup_net_dict)
    # 定义优化器
    optimizer_det = SGD(model.det_net.parameters(), lr=0.001, momentum=0.9)
    optimizer_sup = SGD(model.sup_net.parameters(), lr=0.001, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 训练模型
    for epoch in range(50):
        train_one_epoch(epoch, model, train_loader, optimizer_det, optimizer_sup, device)
        mAP = evaluate_model(model, val_loader, val_dataset.coco)
        if mAP > max_mAP:
            max_mAP = mAP
            torch.save(model.state_dict(), f'./logs/det-sfb_best.pth')
        if epoch >= 3:
            os.remove(f'./logs/det-sfb_{epoch-3}.pth')
        torch.save(model.state_dict(), f'./logs/det-sfb_{epoch}.pth')
