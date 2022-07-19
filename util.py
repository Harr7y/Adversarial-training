import logging
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# cifar10_mean = (0, 0, 0) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
# cifar10_std = (1, 1, 1) # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# for logger
def setup_logging(log_file='log.txt', filemode='w'):
    """
    Setup logging configuration
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%m/%d %I:%M:%S %p')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def set_seed(seed):
    np.random.seed(seed)
    # sets the seed for generating random numbers.
    torch.manual_seed(seed)
    # Sets the seed for generating random numbers for the current GPU.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs.
    # It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    cudnn.deterministic = True
    cudnn.benchmark = True
    cudnn.enabled = True


def lp_hp(temp):
    # 输入是[B, C, H, W]
    if len(temp) == 4:
        temp = torch.reshape(temp, (temp.size(0), temp.size(1), -1))
    x_l = torch.mean(temp, -1, keepdim=True)  # [B, C, 1]
    x_h = temp - x_l  # [B, C, D]
    x_l = temp - x_h  # [B, C, D]

    x_h = torch.norm(x_h, dim=2)  # [B, C,]
    x_l = torch.norm(x_l, dim=2)  # [B, C,]

    ratio = (x_l / x_h).mean(dim=0) #[C,]

    return ratio
