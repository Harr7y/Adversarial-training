import torchvision
import torch
from torchvision import transforms


def get_loaders_mnist(batch_size, data_dir='/home/harry/nnet/NIPS/mnist_data'):
    """
    """
    train_dataset = torchvision.datasets.MNIST(data_dir,
                                          download=True,
                                          train=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            #   transforms.Normalize((0.1307,), (0.3081,))
                                          ]))

    test_dataset = torchvision.datasets.MNIST(data_dir,
                                         download=True,
                                         train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            #  transforms.Normalize((0.1307,), (0.3081,))
                                         ]))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    return train_loader, test_loader

def test():
    data_dir = '/home/harry/nnet/NIPS/mnist_data'
    train_loader, _ = get_loaders_mnist(512, data_dir)
    for X,y in train_loader:
        print(X.shape)
        print(y[:10])
        break

# test()