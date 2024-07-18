from typing import Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


class TransformsPretrain:
    def __init__(self, train: bool, image_size: int = None):

        test_transform = A.Compose(
            [
                (A.LongestMaxSize(max_size=256, interpolation=1)),
                A.PadIfNeeded(
                    min_height=256,
                    min_width=256,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                (
                    A.CenterCrop(height=image_size, width=image_size)
                    if image_size
                    else A.NoOp()
                ),
                ToTensorV2(),
            ]
        )

        train_transform = A.Compose(
            [
                (A.LongestMaxSize(max_size=256, interpolation=1)),
                A.PadIfNeeded(
                    min_height=256,
                    min_width=256,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                (
                    A.RandomCrop(height=image_size, width=image_size)
                    if image_size
                    else A.NoOp()
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.ChannelShuffle(p=0.3),
                A.RGBShift(p=0.3),
                A.ChannelDropout(p=0.3),
                ToTensorV2(),
            ]
        )

        self.transforms = train_transform if train else test_transform

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


class TransformsFinetune:
    def __init__(self, train: bool, image_size: int = None):

        test_transform = A.Compose(
            [
                (A.LongestMaxSize(max_size=256, interpolation=1)),
                A.PadIfNeeded(
                    min_height=256,
                    min_width=256,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                (
                    A.CenterCrop(height=image_size, width=image_size)
                    if image_size
                    else A.NoOp()
                ),
                ToTensorV2(),
            ]
        )

        train_transform = A.Compose(
            [
                (A.LongestMaxSize(max_size=256, interpolation=1)),
                A.PadIfNeeded(
                    min_height=256,
                    min_width=256,
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                (
                    A.RandomCrop(height=image_size, width=image_size)
                    if image_size
                    else A.NoOp()
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                ToTensorV2(),
            ]
        )

        self.transforms = train_transform if train else test_transform

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]
