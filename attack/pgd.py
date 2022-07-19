# https://github.com/P2333/Bag-of-Tricks-for-AT/blob/master/train_cifar.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append('..')
from torch.autograd import Variable

from loss.CW_loss import CW_loss
from loss.DLR_loss import dlr_loss
from DataAugmentation import mixup_data, mixup_criterion

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def pgd_whitebox(model, X, y, epsilon, alpha,
                 attack_iters=20, restarts=1,
                use_CWloss=False, normalize=None):

    model.eval()
    for _ in range(restarts):
        x_adv = X.detach() + torch.empty_like(X).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()

        for _ in range(attack_iters):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if use_CWloss:
                    loss = CW_loss(model(normalize(x_adv)), y)
                else:
                    loss = F.cross_entropy(model(normalize(x_adv)), y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


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


# atk = attack_pgd(model, x, y, 8/255, 2/255, 10, 1)
def attack_pgd_plus(model, X, y, epsilon=8/255, alpha=2/255, attack_iters=10, restarts=1,
               norm='l_inf', mixup=False, y_a=None, y_b=None, lam=None,
               early_stop=False, early_stop_pgd_max=1,
               multitarget=False,
               use_DLRloss=False, use_CWloss=False,
               epoch=0, totalepoch=110, gamma=0.8,
               use_adaptive=False, s_HE=15,
               fast_better=False, BNeval=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    if BNeval:
        model.eval()

    for _ in range(restarts):
        # early stop pgd counter for each x
        early_stop_pgd_count = early_stop_pgd_max * torch.ones(y.shape[0], dtype=torch.int32).cuda()

        # initialize perturbation
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        # for early_stop
        iter_count = torch.zeros(y.shape[0])

        # craft adversarial examples
        for _ in range(attack_iters):
            output = model(normalize(X + delta))

            # if use early stop pgd
            if early_stop:
                # calculate mask for early stop pgd
                if_success_fool = (output.max(1)[1] != y).to(dtype=torch.int32)
                early_stop_pgd_count = early_stop_pgd_count - if_success_fool
                index = torch.where(early_stop_pgd_count > 0)[0]
                iter_count[index] = iter_count[index] + 1
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            # Whether use mixup criterion
            if fast_better:
                loss_ori = F.cross_entropy(output, y)
                grad_ori = torch.autograd.grad(loss_ori, delta, create_graph=True)[0]
                loss_grad = (alpha / 4.) * (torch.norm(grad_ori.view(grad_ori.shape[0], -1), p=2, dim=1) ** 2)
                loss = loss_ori + loss_grad.mean()
                loss.backward()
                grad = delta.grad.detach()

            elif not mixup:
                if multitarget:
                    random_label = torch.randint(low=0, high=10, size=y.shape).cuda()
                    random_direction = 2*((random_label == y).to(dtype=torch.float32) - 0.5)
                    loss = torch.mean(random_direction * F.cross_entropy(output, random_label, reduction='none'))
                    loss.backward()
                    grad = delta.grad.detach()
                elif use_DLRloss:
                    beta_ = gamma * epoch / totalepoch
                    loss = (1. - beta_) * F.cross_entropy(output, y) + beta_ * dlr_loss(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                elif use_CWloss:
                    beta_ = gamma * epoch / totalepoch
                    loss = (1. - beta_) * F.cross_entropy(output, y) + beta_ * CW_loss(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
                else:
                    if use_adaptive:
                        loss = F.cross_entropy(s_HE * output, y)
                    else:
                        loss = nn.CrossEntropyLoss()(output, y)
                        # loss = F.cross_entropy(output, y)
                    loss.backward()
                    grad = delta.grad.detach()
            else:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
                loss.backward()
                grad = delta.grad.detach()


            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    if BNeval:
        model.train()

    return max_delta, iter_count