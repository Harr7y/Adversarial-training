# https://docs.activeloop.ai/datasets/tiny-imagenet-dataset
# import hub
# train_ds = hub.load("hub://activeloop/tiny-imagenet-train")
# # test_ds = hub.load("hub://activeloop/tiny-imagenet-test")
# val_ds = hub.load("hub://activeloop/tiny-imagenet-validation")
#
# train_dataloader = train_ds.pytorch(num_workers=2, batch_size=64, shuffle=True)
# val_dataloader = val_ds.pytorch(num_workers=2, batch_size=64, shuffle=False)

# https://github.com/pranavphoenix/TinyImageNetLoader/blob/main/tinyimagenetloader.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import os, glob
from torchvision.io import read_image


in_mean = (0.4802, 0.4481, 0.3975)
in_std = (0.2770, 0.2691, 0.2821)


def get_loaders_tin(batch_size, dir=None):
    if dir:
        data_dir = dir
    else:
        data_dir = '/home/mnt/datasets/tiny-imagenet-200'

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(in_mean, in_std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(in_mean, in_std),
    ])
    train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,
                                             shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size,
                                             shuffle=False, num_workers=2, pin_memory=True)

    return train_dataloader, val_dataloader


def test():
    train_loader, val_loader = get_loaders_tin(4)
    for i, (X,y) in enumerate(val_dataloader):
        print(X.shape)
        print(y)
        break

# test()


# batch_size = 64
#
# id_dict = {}
# for i, line in enumerate(open('/home/mnt/datasets/tiny-imagenet-200/wnids.txt', 'r')):
#     id_dict[line.replace('\n', '')] = i
#
#
# class TrainTinyImageNetDataset(Dataset):
#     def __init__(self, id, transform=None):
#         self.filenames = glob.glob("/home/mnt/datasets/tiny-imagenet-200/train/*/*/*.JPEG")
#         self.transform = transform
#         self.id_dict = id
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path)
#         label = self.id_dict[img_path.split('/')[4]]
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#         return image, label
#
#
# class TestTinyImageNetDataset(Dataset):
#     def __init__(self, id, transform=None):
#         self.filenames = glob.glob("/home/mnt/datasets/tiny-imagenet-200/val/images/*.JPEG")
#         self.transform = transform
#         self.id_dict = id
#         self.cls_dic = {}
#         for i, line in enumerate(open('/home/mnt/datasets/tiny-imagenet-200/val/val_annotations.txt', 'r')):
#             a = line.split('\t')
#             img, cls_id = a[0], a[1]
#             self.cls_dic[img] = self.id_dict[cls_id]
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         img_path = self.filenames[idx]
#         image = read_image(img_path)
#         label = self.cls_dic[img_path.split('/')[-1]]
#         if self.transform:
#             image = self.transform(image.type(torch.FloatTensor))
#         return image, label



