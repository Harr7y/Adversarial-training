import torch
import torch.nn
import time
from torch.autograd import Variable
from util import AverageMeter, accuracy, normalize
from attack.pgd import attack_pgd, clamp, pgd_whitebox
from tqdm import tqdm
import torch.nn as nn
from loss.orthogonalconv import deconv_orth_dist, orth_dist

def train_epoch(args, epoch, dataloader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(tqdm(dataloader)):

        input = input.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(normalize(input))
        loss = criterion(output_clean, target)
        if args.loss == 'ols':
            criterion.accumulate(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def train_epoch_adv(args, epoch, model, dataloader, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()

        # generate Adversarial Examples (AEs)
        if args.norm == 'l_inf':
            X_pgd = pgd_whitebox(model, input, target, epsilon=args.train_epsilon,
                               alpha=args.train_alpha, attack_iters=args.train_iters,
                                  restarts=1, use_CWloss=False, normalize=normalize)
        model.train()
        # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # compute output
        adv_example = normalize(X_pgd)

        optimizer.zero_grad()
        output_ae = model(adv_example)
        if args.loss in ['ce', 'ls']:
            loss = criterion(output_ae, target)

        elif args.loss == 'ols':
            output_clean = model(normalize(input))
            loss = 0.5 * criterion(output_ae, target) + \
                   0.5 * nn.CrossEntropyLoss()(output_ae, target)
            criterion.accumulate(output_clean, target)

        elif args.loss == 'ols2':
            output_clean = model(normalize(input))
            loss = 0.5 * criterion(output_clean, output_ae, target) + \
                   0.5 * nn.CrossEntropyLoss()(output_ae, target)

        # extra loss
        if args.orthogonal:
            #####
            diff = orth_dist(model.module.layer2[0].shortcut[0].weight) + \
                   orth_dist(model.module.layer3[0].shortcut[0].weight) + \
                   orth_dist(model.module.layer4[0].shortcut[0].weight)
            diff += deconv_orth_dist(model.module.layer1[0].conv1.weight, stride=1) + deconv_orth_dist(
                model.module.layer1[1].conv1.weight, stride=1)
            diff += deconv_orth_dist(model.module.layer2[0].conv1.weight, stride=2) + deconv_orth_dist(
                model.module.layer2[1].conv1.weight, stride=1)
            diff += deconv_orth_dist(model.module.layer3[0].conv1.weight, stride=2) + deconv_orth_dist(
                model.module.layer3[1].conv1.weight, stride=1)
            diff += deconv_orth_dist(model.module.layer4[0].conv1.weight, stride=2) + deconv_orth_dist(
                model.module.layer4[1].conv1.weight, stride=1)
            #####
            loss += args.orth_rate * diff

        if args.fre_loss:
            output_clean = model(normalize(input))
            # adv_fft = torch.rfft(output_ae, signal_ndim=1, normalized=False, onesided=False)
            # clean_fft = torch.rfft(output_clean, signal_ndim=1, normalized=False, onesided=False)

            adv_fft = torch.rfft(output_ae, signal_ndim=2, normalized=False, onesided=False)
            clean_fft = torch.rfft(output_clean, signal_ndim=2, normalized=False, onesided=False)

            fre_loss = torch.nn.L1Loss()(adv_fft, clean_fft)
            loss += args.fre_rate * fre_loss

        loss.backward()
        optimizer.step()

        output_ae = output_ae.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_ae.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg
