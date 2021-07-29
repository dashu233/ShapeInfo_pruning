import argparse
import os
import random
import shutil
import time
import warnings
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from build_loader import build_data_loader
from model_builder import build_model, model_to_gpu
from utils import is_main_process,accuracy,adjust_learning_rate,ProgressMeter,AverageMeter,channel_remaining
import torchvision.models as models
from prune_method import prune_model
import torch.nn.utils.prune as prune

import json

from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
def add_dcp_parameters():
    parser.add_argument('--dcp_loss_penalty', type=float, default=1e-5)
    parser.add_argument('--dcp_name_list', type=str,
                        default="[\"conv1\",\"layer1.0.conv1\",\"layer2.0.conv1\",\"layer3.0.conv1\"]")
    parser.add_argument('--dcp_prune_criterion_iter', type=int, default=100)
    parser.add_argument('--dcp_epoch_per_stage',type=int,default=20)
    parser.add_argument('--dcp_criterion_finetune_epoch',type=int,default=1)
def add_st_gRDA_parameters():
    parser.add_argument('--st_gRDA_mu',type=float,default=0.501)
    parser.add_argument('--st_gRDA_c',type=float,default=1e-4)
def add_scb_parameters():
    parser.add_argument('--scb_slimming_penalty',type=float,default=1e-4)
    parser.add_argument('--scb_loss_penalty',type=float,default=0.1)


parser.add_argument('--prune',type=bool,default=False,help='whether prune or not')
parser.add_argument('--prune_steps',type=str,default='[100,150,200,250]',help='prune at step i')
parser.add_argument('--method',type=str, default='network_slimming',help='criterion for pruning')
parser.add_argument('--slimming_penalty',type=float,default=1e-3,help='penalty of network slimming term')
parser.add_argument('--prune_rate',type=float,default=0.5,help='final pruning rate')
parser.add_argument('--logdir',type=str,default='log',help='log file dir')
parser.add_argument('--dataset',type=str, default='ImageNet')
parser.add_argument('--expname',type=str,default='exp1')
parser.add_argument('--checkpoint_interval',type=int,default=20,help='epoch interval for saving checkpoint')

parser.add_argument('--lr_adjust_steps',type=str,default='[30,60,90]',help='lr *= 0.1 step')
parser.add_argument('--vflip',type=bool,default=False)
parser.add_argument('--hflip',type=bool,default=True)
parser.add_argument('--view',type=bool,default=False)

add_dcp_parameters()
add_st_gRDA_parameters()
add_scb_parameters()
args = parser.parse_args()
best_acc1 = 0



def main():
    args.prune_steps = json.loads(args.prune_steps)
    args.lr_adjust_steps = json.loads(args.lr_adjust_steps)
    args.dcp_name_list = json.loads(args.dcp_name_list)

    if not os.path.exists('output'):
        os.mkdir('output')

    if os.path.exists(os.path.join('output',args.expname)):
        print('warning: An exist exp dir, type exit/e to exit python or anything else to continue')
        ch = input()
        if ch == 'exit' or ch == 'e':
            exit()
    else:
        os.mkdir(os.path.join('output',args.expname))

    args.logdir = os.path.join('output',args.expname,args.logdir)
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.view:
        model,aux_model,learnable_keys = build_model(args)
        model_to_gpu(args,model,aux_model,learnable_keys)

        args.resume = 'output/cifar10_st_gRDA_pretrained/ep099.pth.tar'

        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
                # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)

        model.load_state_dict(checkpoint['state_dict'])

        channel_remaining(model)
        return


    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    if is_main_process(args):
        writer = SummaryWriter(args.logdir)
    else:
        writer = None
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    model,aux_model,leanable_keys,handles = build_model(args)
    # aux_model is a dict contain different part for different method

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    para_group = [{'params':aux_model[ky].parameters()} for ky in leanable_keys]
    para_group.append({'params':model.parameters()})

    optimizer = torch.optim.SGD(para_group, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.method == 'DCP':
        args.all_layer_names = []
        for name,m in model.named_modules():
            args.all_layer_names.append(name)
    
    model_to_gpu(args,model,aux_model,leanable_keys)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    val_loader,train_loader,train_sampler = build_data_loader(args)

    prune_model(train_loader,train_sampler,val_loader,model,
        criterion,optimizer,args,writer,aux_model)

    torch.save(model,'final.pth')

if __name__ == '__main__':
    main()
