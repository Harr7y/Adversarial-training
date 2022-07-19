import torch


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]

def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        b = input.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))



# https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py
# def bn_update(loader, model, device=None):
#     r"""Updates BatchNorm running_mean, running_var buffers in the model.
#     It performs one pass over data in `loader` to estimate the activation
#     statistics for BatchNorm layers in the model.
#     Args:
#         loader (torch.utils.data.DataLoader): dataset loader to compute the
#             activation statistics on. Each data batch should be either a
#             tensor, or a list/tuple whose first element is a tensor
#             containing data.
#         model (torch.nn.Module): model for which we seek to update BatchNorm
#             statistics.
#         device (torch.device, optional): If set, data will be trasferred to
#             :attr:`device` before being passed into :attr:`model`.
#     """
#     if not _check_bn(model):
#         return
#     was_training = model.training
#     model.train()
#     momenta = {}
#     model.apply(_reset_bn)
#     model.apply(lambda module: _get_momenta(module, momenta))
#     n = 0
#     for input in loader:
#         if isinstance(input, (list, tuple)):
#             input = input[0].cuda()
#         b = input.size(0)
#
#         momentum = b / float(n + b)
#         for module in momenta.keys():
#             module.momentum = momentum
#
#         if device is not None:
#             input = input.to(device)
#
#         model(input)
#         n += b
#
#     model.apply(lambda module: _set_momenta(module, momenta))
#     model.train(was_training)
#
#
# # BatchNorm utils
# def _check_bn_apply(module, flag):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         flag[0] = True
#
#
# def _check_bn(model):
#     flag = [False]
#     model.apply(lambda module: _check_bn_apply(module, flag))
#     return flag[0]
#
#
# def _reset_bn(module):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.running_mean = torch.zeros_like(module.running_mean)
#         module.running_var = torch.ones_like(module.running_var)
#
#
# def _get_momenta(module, momenta):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         momenta[module] = module.momentum
#
#
# def _set_momenta(module, momenta):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.momentum = momenta[module]
