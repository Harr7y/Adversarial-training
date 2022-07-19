import torch
import torch.nn
from util import AverageMeter, accuracy, normalize
from attack.pgd import attack_pgd, clamp, pgd_whitebox
from tqdm import tqdm


def eval_clean(args, epoch, dataloader, model):
    top1 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(normalize(input))
        output_clean = output_clean.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_clean.data, target)[0]

        top1.update(prec1.item(), input.size(0))

    # print('eval_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def eval_pgd(args, epoch, dataloader, model):
    top1 = AverageMeter()
    model.eval()

    for i, (input, target) in enumerate(tqdm(dataloader)):
        input = input.cuda()
        target = target.cuda()

        # generate Adversarial Examples (AEs)
        if args.norm == 'l_inf':
            X_pgd = pgd_whitebox(model, input, target, epsilon=args.test_epsilon,
                               alpha=args.test_alpha, attack_iters=args.test_iters,
                                  restarts=1, use_CWloss=False, normalize=normalize)


            # delta = attack_pgd(model, input, target, epsilon=args.test_epsilon,
            #                    alpha=args.test_alpha, attack_iters=args.test_iters,
            #                       restarts=1, use_CWloss=False, normalize=normalize)
        # adv_example = normalize(torch.clamp(input + delta[:input.size(0)], min=0.0, max=1.0))

        model.eval()
        # compute output
        output_ae = model(normalize(X_pgd))

        output_ae = output_ae.float()
        # measure accuracy and record loss
        prec1 = accuracy(output_ae.data, target)[0]

        top1.update(prec1.item(), input.size(0))

    # print('eval_pgd20 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
