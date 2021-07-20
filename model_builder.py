import custom_model

import torch.utils.data.distributed
import torchvision.models as models
import copy
import torch.nn.utils.prune as prune
from custom_model import DCPClassifier

def need_featmap(args):
    return args.method in ['DCP']

def add_prune_mask(model,args):
    for name, module in model.named_modules():
        if args.method == 'slimming':
            if isinstance(module, torch.nn.BatchNorm2d):
                prune.l1_unstructured(module, name='weight', amount=0)
                prune.l1_unstructured(module, name='bias', amount=0)
        elif args.method == 'DCP':
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0)
                # prune.l1_unstructured(module, name='bias', amount=0)
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0)
                prune.l1_unstructured(module, name='bias', amount=0)


def build_model(args):
    aux_model = {}
    learnable_keys = []
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch in models.__dict__:
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print('find {} in custom_model'.format(args.arch))
            if args.arch in custom_model.__dict__:
                model = custom_model.__dict__[args.arch](pretrained=True)
            else:
                print('undefined arch')
                raise NotImplementedError

    else:
        if args.arch in models.__dict__:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()
        else:
            print('find {} in custom_model'.format(args.arch))
            if args.arch in custom_model.__dict__:
                model = custom_model.__dict__[args.arch]()
            else:
                print('undefined arch')
                raise NotImplementedError

    if args.method == 'DCP':
        name_list = args.feat_name_list
        #name_list = ['conv1','layer1.0.conv1','layer2.0.conv1','layer3.0.conv1']
        def gethook(tarlist,name):
            def hook(model, input, output):
                tarlist[name] = output.clone()
            return hook

        aux_model['proto_model'] = copy.deepcopy(model)
        aux_model['now_feats'] = {}
        for name,m in model.named_modules():
            if name in name_list:
                hk = gethook(aux_model['now_feats'],name)
                m.register_forward_hook(hk)

        aux_model['orig_feats'] = {}
        for name,m in aux_model['proto_model'].named_modules():
            if name in name_list:
                hk = gethook(aux_model['orig_feats'],name)
                m.register_forward_hook(hk)

    if args.method == "DCP":
        if args.dataset == 'Cifar10':
            aux_model['dcp_classifier'] = DCPClassifier(args,model)
            learnable_keys.append('dcp_classifier')

    ngpus_per_node = torch.cuda.device_count()
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            for ky in aux_model:
                if isinstance(aux_model[ky],torch.nn.Module):
                    aux_model[ky].cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)


            if args.prune:
                add_prune_mask(model,args)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            for ky in aux_model:
                if isinstance(aux_model[ky],torch.nn.Module):
                    aux_model[ky] = torch.nn.parallel.DistributedDataParallel(aux_model[ky], device_ids=[args.gpu])
        else:
            model.cuda()
            for ky in aux_model:
                if isinstance(aux_model[ky], torch.nn.Module):
                    aux_model[ky].cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set

            if args.prune:
                add_prune_mask(model,args)

            model = torch.nn.parallel.DistributedDataParallel(model)
            for ky in aux_model:
                if isinstance(aux_model[ky], torch.nn.Module):
                    aux_model[ky] = torch.nn.parallel.DistributedDataParallel(aux_model[ky])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        for ky in aux_model:
            if isinstance(aux_model[ky], torch.nn.Module):
                aux_model[ky].cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()

        else:
            model = torch.nn.DataParallel(model).cuda()
        for ky in aux_model:
            if isinstance(aux_model[ky], torch.nn.Module):
                aux_model[ky] = torch.nn.DataParallel(model).cuda()

    return model,aux_model,learnable_keys