import torchvision
import torch
from torchvision import transforms


def load_svhn_data(batch_size, data_dir='/home/harry/nnet/NIPS/svhn_data'):
    """
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def test():
    data_dir = '/home/harry/nnet/NIPS/mnist_data'
    train_loader, _ = load_svhn_data(64, data_dir)
    for X,y in train_loader:
        print(X.shape)
        print(y[:10])
        break

# test()