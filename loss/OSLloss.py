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


class OnlineLabelSmoothing(nn.Module):
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


    def forward(self, logits, target):
        # logits: logits are the outputs of the network before softmax.
        target = target.view(-1,)
        log_probs = torch.log_softmax(logits, dim=-1)

        # softlabel : [batch size, num_classes]
        softlabel = self.matrix[target]
        loss = (- softlabel * log_probs).sum(dim=-1)

        return loss.mean()


    def accumulate(self, logits, target):
        with torch.no_grad():
            logits = torch.softmax(logits.detach(), dim=1)
            # sort_args: indexes in descending order
            sort_args = torch.argsort(logits, dim=1, descending=True)
            # accumulate correct predictions
            for k in range(target.shape[0]):
                # only accumulate correct predictions
                if target[k] != sort_args[k, 0]:
                    continue
                self.accumulator[target[k]] += logits[k]
                self.count[target[k]] += 1

        # # accumulate correct predictions
        # prob = torch.softmax(logits.detach(), dim=1)
        # _, pred = torch.max(prob, 1)
        # # right prediction
        # correct_index = pred.eq(target)
        # correct_p = prob[correct_index]
        # correct_label = target[correct_index].tolist()
        #
        # # accumulator
        # # modification
        # for i in range(len(correct_label)):
        #     self.accumulator[correct_label[i]] += correct_p[i]
        # # origin: Wrong
        # #self.accumulator[correct_label] += correct_p
        #
        # for k, v in Counter(correct_label).items():
        #     self.count[k] += v



    def update(self):
        index = torch.where(self.count > 0)[0]
        self.accumulator[index] = self.accumulator[index] / self.count[index]
        # reset matrix and update
        nn.init.constant_(self.matrix, 0.)

        # not necessary I think (since the sum of each column is 1)
        norm = self.accumulator.sum(dim=1).view(-1, 1)
        index = torch.where(norm > 0)[0]
        self.matrix[index] = self.accumulator[index] / norm[index]

        # reset
        nn.init.constant_(self.accumulator, 0.)
        nn.init.constant_(self.count, 0.)


if __name__ == '__main__':
    import random
    ols = OnlineLabelSmoothing(num_classes=5, use_gpu=False)
    x = torch.eye(5)
    x[4][1]=0.8
    x[4][4]=0.2
    y = torch.LongTensor([0, 1, 2, 3, 1])

    l = ols(x, y)
    ols.accumulate(x, y)
    print('ols:', l)
    ols.update()
    print(ols.matrix)

    l = ols(x, y)
    print('ols:', l)