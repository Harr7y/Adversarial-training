import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
from resnet_feature import ResNet18
from resnet_nlm2 import ResNet18 as Res_nlm
# from resnet_gf import ResNet18 as Res_gf

sys.path.insert(0, '..')

from resnet import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

upper_limit, lower_limit = 1, 0
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=False, normalize=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            if use_CWloss:
                loss = CW_loss(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts=1, eps=8, step=2, use_CWloss=False, normalize=None):
    epsilon = eps / 255.
    alpha = step / 255.
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, use_CWloss=use_CWloss,
                               normalize=normalize)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/harry/dataset/cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nlm', type=int, default=0)


    args = parser.parse_args()

    # load model
    # model = Res_nlm(gf=4)
    model = Res_nlm(args.nlm, 200)
    target_model_path = '/home/harry/nnet/MART/ResNet18_nlm_tin/' + str(args.nlm) +'-nlm-best.pt'
    target_state_dict = torch.load(target_model_path)
    newckpt = {}
    for k, v in target_state_dict.items():
        if "module." in k:
            single_k = k.replace('module.', '')
            newckpt[single_k] = v
        else:
            newckpt[k] = v
    del target_state_dict
    model.load_state_dict(newckpt, strict=False)

    model.cuda()
    model.eval()

    # load data
    # Cifar10
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=8)

    # Tint-ImageNet
    # data_dir = '/home/mnt/datasets/tiny-imagenet-200'
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize(in_mean, in_std),
    # ])
    # val_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    # test_loader = torch.utils.data.DataLoader(val_datasets, batch_size=128,
    #                                           shuffle=False, num_workers=8, pin_memory=True)

    # load attack
    loss, cw_acc = evaluate_pgd(test_loader, model, attack_iters=30, restarts=1, eps=8, step=2, use_CWloss=True, normalize=normalize)
    print('CW Test Accuracy: {:.2f}%'.format(100. * cw_acc))