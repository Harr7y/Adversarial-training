import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import os, glob
from torchvision.io import read_image
import numpy as np

in_mean = (0.485, 0.456, 0.406)
in_std = (0.229, 0.224, 0.225)



def get_loaders_in(batch_size, dir=None):
    """
    Can load both original data and adversarial data
    """
    if dir:
        data_dir = dir
    else:
        # 3090
        data_dir = '/home/mnt/datasets/ImageNet2012'
        # v100
        # data_dir = '/mnt/nfsdisk/data/imagenet'

    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")
    scale_size = 224

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(scale_size),
            transforms.ToTensor(),
            # transforms.GaussianBlur(7, sigma=4),
            # transforms.Normalize(in_mean, in_std),
        ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(scale_size),
        transforms.ToTensor(),
        # transforms.GaussianBlur(7, sigma=4),
        # transforms.Normalize(in_mean, in_std),
    ])

    train_datasets = torchvision.datasets.ImageFolder(traindir, train_transform)

    val_datasets = torchvision.datasets.ImageFolder(valdir, val_transform)

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)

    return train_dataloader, val_dataloader


def test():
    train_loader, val_loader = get_loaders_in(10)
    for i, (X,y) in enumerate(train_loader):
        print(X.shape)
        print(y)
        np.save('imagenet.npy', X.numpy())
        break

# test()


