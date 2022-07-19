import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

# norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


# CIFAR-10 dataloader
def get_loaders_cifar10(dir_, batch_size, norm=False):
    if norm:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    num_workers = 8
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, #//4,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


# # For CAT
# class Cat_dataloader(Dataset):
#     def __init__(self, data, is_train=True, transform=None):
#         self.data = datasets.__dict__[data.upper()]('../cifar_data', train=is_train, download=True, transform=transform)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, i):
#         image = self.data[i][0]
#         label = self.data[i][1]
#         index = i
#         return image, label, index
#
# batch_size = 256
# train_transforms = transforms.Compose([
# transforms.RandomCrop(32, padding=4),
# transforms.RandomHorizontalFlip(),
# transforms.ToTensor()])
# train_dataset = Cat_dataloader('cifar10', is_train=True, transform=train_transforms)
# test_dataset = Cat_dataloader('cifar10', is_train=False, transform=transforms.ToTensor())
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
#                           shuffle=True, drop_last=False, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
#                          shuffle=False, drop_last=False, num_workers=4)


if __name__ == '__main__':
    dir = '/home/harry/dataset/cifar10'
    train_loader, test_loader = get_loaders_cifar10(dir, 256)
    print(len(test_loader))
    print(len(test_loader.dataset))

    for i, data in enumerate(train_loader):
        print(data[0].shape)
        print(data[1].shape)

        if i == 1:
            break