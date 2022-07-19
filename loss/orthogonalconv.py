import torch
import numpy as np
from torch.nn import functional as F

def conv_orth_dist(kernel, stride = 1):
    [o_c, i_c, w, h] = kernel.shape
    assert (w == h) ,"Do not support rectangular kernel"
    # half = np.floor(w/2)
    assert stride <w ,"Please use matrix orthgonality instead"
    new_s = stride *( w -1) +  w  # np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s * new_s * i_c).reshape((new_s * new_s * i_c, i_c, new_s, new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s * new_s * i_c, -1))
    Vmat = out[np.floor(new_s ** 2 / 2).astype(int)::new_s ** 2, :]
    temp = np.zeros((i_c, i_c * new_s ** 2))
    for i in range(temp.shape[0]): temp[i, np.floor(new_s ** 2 / 2).astype(int) + new_s ** 2 * i] = 1
    return torch.norm(Vmat @ torch.t(out) - torch.from_numpy(temp).float().cuda())


def deconv_orth_dist(kernel, stride=2, padding=1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1] / 2))
    target[:, :, ct, ct] = torch.eye(o_c).cuda()
    return torch.norm(output - target)


def orth_dist(mat, stride=None):
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1, 0)
    return torch.norm(torch.t(mat) @ mat - torch.eye(mat.shape[1]).cuda())