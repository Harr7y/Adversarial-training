# Online Soft Label loss
# learned from https://github.com/zhangchbin/OnlineLabelSmoothing/blob/main/cifar/scripts/train_cifar_all_methods.py
# https://github.com/Kurumi233/OnlineLabelSmoothing/blob/5cb9f1acd3c0507fa175ff8de04898a4b0d1d86a/models/OLS.py

"""
Reference:
    paper: Delving Deep into Label Smoothing.
"""
import torch
import torch.nn as nn
from collections import Counter


class OSL2(nn.Module):
    def __init__(self, num_classes=10, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        # matrix: store the target probability values learned from the last epoch
        # row: num of classes;  column: record the probability values
        # Normal CE loss, matrix is an eye matrix
        self.matrix = torch.zeros((num_classes, num_classes))
        self.matrix.requires_grad = False
        # accumulator: record the probability values accumulated for each class within an epoch
        self.accumulator = torch.zeros((num_classes, num_classes))
        # count: record the cumulative number of each class (successful prediction)
        self.count = torch.zeros((num_classes, 1))
        if use_gpu:
            self.matrix = self.matrix.cuda()
            self.accumulator = self.accumulator.cuda()
            self.count = self.count.cuda()


    def forward(self, logits_clean, logits_adv, target):
        # logits: logits are the outputs of the network before softmax.
        # reset
        nn.init.constant_(self.accumulator, 0.)
        nn.init.constant_(self.count, 0.)
        nn.init.constant_(self.matrix, 0.)

        with torch.no_grad():
            probs = torch.softmax(logits_clean.detach(), dim=1)
            # sort_args: indexes in descending order
            sort_args = torch.argsort(probs, dim=1, descending=True)
            # accumulate correct predictions
            for k in range(target.shape[0]):
                # only accumulate correct predictions
                if target[k] != sort_args[k, 0]:
                    continue
                self.accumulator[target[k]] += probs[k]
                self.count[target[k]] += 1

        index = torch.where(self.count > 0)[0]
        self.accumulator[index] = self.accumulator[index] / self.count[index]
        # not necessary I think (since the sum of each column is 1)
        norm = self.accumulator.sum(dim=1).view(-1, 1)
        index = torch.where(norm > 0)[0]
        self.matrix[index] = self.accumulator[index] / norm[index]

        # 没有正确预测分类的，设置为hard label
        index_zero = torch.where(self.count == 0.0)[0]
        self.matrix[index_zero, index_zero] = 1

        # 计算loss
        target = target.view(-1,)
        log_probs = torch.log_softmax(logits_adv, dim=-1)

        # softlabel : [batch size, num_classes]
        softlabel = self.matrix[target]
        loss = (- softlabel * log_probs).sum(dim=-1)

        return loss.mean()


if __name__ == '__main__':
    import random
    ols = OSL2(num_classes=5, use_gpu=False)
    x = torch.eye(5)
    x[4][1]=0.8
    x[4][4]=0.2
    y = torch.LongTensor([0, 1, 2, 3, 1])

    l = ols(x, x, y)

    print('ols:', l)

    print(ols.matrix)

    l = ols(x, x, y)
    print('ols:', l)