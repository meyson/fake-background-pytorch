import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import segmentation_models_pytorch as smp
# FIXME
from utils import *
from datasets.default_augmentations import *
from datasets import CocoSegnentation

ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['person']
# TODO
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
EPOCH = 40
BATCH_SIZE = 1

if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True


def main():
    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
    )

    print(model)
    print(f'N of parameters = {(count_parameters(model) / 10**6):.2f}m')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)

    # augmentations
    train_augs = get_validation_augmentation()
    valid_augs = get_validation_augmentation()

    coco_root = '/mnt/xpg/datasets/fast-ai-coco/'
    coco_ann = '/mnt/xpg/datasets/fast-ai-coco/annotations/'
    coco_ann_train = coco_ann + 'instances_train2017.json'
    coco_ann_valid = coco_ann + 'instances_val2017.json'

    train_dataset = CocoSegnentation(coco_root + 'train2017', coco_ann, coco_ann_train,
                                     augmentation=train_augs, preprocessing=preprocessing)
    valid_dataset = CocoSegnentation(coco_root + 'val2017', coco_ann, coco_ann_valid,
                                     augmentation=valid_augs, preprocessing=preprocessing)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=4)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.001),
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=5)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs
    max_score = 0
    for i in range(0, 5):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model.state_dict(), 'saved/best_model.pth')
            print('Model saved!')

            optimizer.param_groups[0]['lr'] = 1e-4

        # scheduler.step(valid_logs['iou_score'])


if __name__ == '__main__':
    main()
