from glob import glob
from random import shuffle
from urllib.request import urlopen

import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


class SortedListDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(SortedListDataset, self).__init__()
        self.img_list = sorted(glob(root + "/*.JPEG"))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert(
            "RGB"
        )  # Load image and convert to RGB

        return self.transform(img)


class ShuffledListDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(ShuffledListDataset, self).__init__()
        self.img_list = glob(root + "/*.JPEG")
        shuffle(self.img_list)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert(
            "RGB"
        )  # Load image and convert to RGB

        return self.transform(img)


def load_imagenet_1k(
    data_path,
    transform_train,
    transform_test,
    batch_size,
    num_workers,
):

    # Load the ImageNet train dataset
    train_dataset = ShuffledListDataset(
        root=data_path + "train",
        transform=transform_train,
    )
    # Load the ImageNet validation dataset
    val_dataset = ShuffledListDataset(
        root=data_path + "val",
        transform=transform_test,
    )
    test_dataset = SortedListDataset(
        root=data_path + "test",
        transform=transform_test,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
