import torch
import torch.distributed as dist
import math
import torch.nn as nn
def is_main_process(args):
    if torch.distributed.is_initialized():
        return dist.get_rank() % args.world_size == 0
    else:
        return True

def channel_remaining(model,mask_way='bn'):
    if mask_way=='bn':
        last_name = ''
        kernel_size = 3
        for name,n in model.named_modules():
            # print(type(n))
            if isinstance(n, nn.Conv2d):
                last_name = name
                kernel_size = n.kernel_size
            if isinstance(n, nn.BatchNorm2d):
                print('conv_name:{},remain channels:{},kernel_size:{}'.format(
                    last_name,int(torch.sum(n.weight_mask).item()),kernel_size))
    return

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        #print(self.meters)
        for i,mt in enumerate(self.meters):
            print(str(mt))
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args,writer=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_steps = args.lr_adjust_steps
    if args.method == 'st_gRDA':
        if writer is not None:
            writer.add_scalar('lr', args.lr, epoch)
        return

    if epoch in adjust_steps:
        args.lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    if writer is not None:
        writer.add_scalar('lr', args.lr, epoch)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
