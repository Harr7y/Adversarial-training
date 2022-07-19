from __future__ import print_function
import os
import time
import torch
import logging
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from models.WRN import WideResNet
from models.ResNet import ResNet18
from models.Preactivate_ResNet import PreActResNet18


from loss.OSLloss import OnlineLabelSmoothing
from loss.LabelSmooth import LabelSmoothingLoss
from loss.OSLloss_tmp import OSL2

from util import setup_logging, set_seed
from eval import eval_clean, eval_pgd
from train import train_epoch, train_epoch_adv

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
# training hyperparameters
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr_drop', type=str, default='75, 90', metavar='LR',
                    help='learning rate drop epoch')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--l2', type=bool, default=False, help='L2 norm penalty')

# PGD hyperparameters
parser.add_argument('--train_epsilon', default=8, help='perturbation')
parser.add_argument('--train_iters', default=10, help='perturb number of steps')
parser.add_argument('--train_alpha', default=2, help='perturb step size')

parser.add_argument('--test_epsilon', default=8, help='perturbation')
parser.add_argument('--test_iters', default=20, help='perturb number of steps')
parser.add_argument('--test_alpha', default=2, help='perturb step size')
parser.add_argument('--norm', default='l_inf', help='perturb style')
# loss
parser.add_argument('--loss', default='ce', choices=['ce', 'ols','ls', 'ols2'], help='different types of loss function')
parser.add_argument('--labelsmooth', default=0.2, type=float, help='label smooth value')
parser.add_argument('--orthogonal', default=False, type=bool, help='orthogonal convolutional layer loss')
parser.add_argument('--orth_rate', default=0.1, type=float, help='rate of difference loss')


parser.add_argument('--model', default='resnet', choices=['resnet', 'wrn', 'preresnet'],
                    help='directory of model for saving checkpoint')
parser.add_argument('--data_path', default='/home/harry/dataset/cifar10',
                    help='directory of cifar10 dataset')
parser.add_argument('--tb_dir', type=str, default='./tb/', help='the tensorboard log')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--trial', type=int, default=0, help='the trial index')



args = parser.parse_args()
args.train_alpha = args.train_alpha / 255.0
args.train_epsilon= args.train_epsilon / 255.0
args.test_alpha = args.test_alpha / 255.0
args.test_epsilon= args.test_epsilon / 255.0

# settings
model_dir = 'ckpt/' + args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logfile = os.path.join(log_dir, str(args.trial) + '.txt')
# initialize the logging
setup_logging(logfile)

logging.info(args)

if not os.path.exists(args.tb_dir):
    os.mkdir(args.tb_dir)
tb_path = args.tb_dir + str(args.trial)
writer = SummaryWriter(tb_path)

set_seed(args.seed)
lr_drop = list(map(int, args.lr_drop.split(',')))

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
# kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr_drop1, lr_drop2 = lr_drop
    lr = args.lr
    if epoch >= lr_drop2:
        lr = args.lr * 0.01
    elif epoch >= lr_drop1:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main():
    if args.model =='resnet':
        model = ResNet18()
    elif args.model == 'preresnet':
        model = PreActResNet18()
    elif args.model == 'wrn':
        model = WideResNet()
    model = nn.DataParallel(model)
    model = model.cuda()

    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay':0.0005},
                  {'params': no_decay, 'weight_decay': 0}]
    else:
        params = model.parameters()
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ols':
        criterion = OnlineLabelSmoothing()
    elif args.loss == 'ls':
        criterion = LabelSmoothingLoss(smoothing=args.labelsmooth)
    elif args.loss == 'ols2':
        criterion = OSL2()

    logging.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

    best_pgd_acc = 0
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        start_time = time.time()

        # adversarial training
        train_acc1, train_loss = train_epoch_adv(args, epoch, model, train_loader, criterion, optimizer)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logging.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                     epoch, time.time() - start_time, lr, train_loss, train_acc1)

        writer.add_scalar('Train/accuracy', train_acc1, epoch)
        writer.add_scalar('Train/loss', train_loss, epoch)
        print('================================================================')

        eval_clean_acc1 = eval_clean(args, epoch, test_loader, model)
        eval_pgd_acc1 = eval_pgd(args, epoch, test_loader, model)
        logging.info('Eval accuracy: \t %.4f, Eval robustness: \t %.4f', eval_clean_acc1, eval_pgd_acc1)

        writer.add_scalar('Eval/accuracy', eval_clean_acc1, epoch)
        writer.add_scalar('Eval/robustness', eval_pgd_acc1, epoch)

        if eval_pgd_acc1 > best_pgd_acc:
            best_pgd_acc = eval_pgd_acc1
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'best_model_' + str(args.trial) +'.pt'))

        print('using time:', time.time() - start_time)

        if args.loss == 'ols':
            criterion.update()

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
