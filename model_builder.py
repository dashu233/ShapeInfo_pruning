from typing import List, Tuple
import custom_model

import torch.utils.data.distributed
import torchvision.models as models
import copy
import torch.nn.utils.prune as prune
from custom_model import DCPClassifier,BasicBlock
import torch.nn as nn

def add_prune_mask(model,args):
    print('add prune mask to model')
    for name, module in model.named_modules():
        if args.method == 'slimming' or 'st_gRDA' or 'st_margin':
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


def build_model(args) -> Tuple[nn.Module,dict,List,List]:
    aux_model = {}
    learnable_keys = []
    handles = []
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch in models.__dict__:
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print('find {} in custom_model'.format(args.arch))
            if args.arch in custom_model.__dict__:
                model = custom_model.__dict__[args.arch](args=args,pretrained=True)
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
                model = custom_model.__dict__[args.arch](args=args)
            else:
                print('undefined arch')
                raise NotImplementedError

    if args.method == 'DCP':
        name_list = args.dcp_name_list
        #name_list = ['conv1','layer1.0.conv1','layer2.0.conv1','layer3.0.conv1']
        def gethook(tarlist,name):
            def hook(model, input, output):
                tarlist[name] = output
            return hook

        aux_model['proto_model'] = copy.deepcopy(model)
        aux_model['now_feats'] = {}
        for name,m in model.named_modules():
            if name in name_list:
                hk = gethook(aux_model['now_feats'],name)
                handles.append(hk)
                m.register_forward_hook(hk)

        aux_model['orig_feats'] = {}
        for name,m in aux_model['proto_model'].named_modules():
            if name in name_list:
                hk = gethook(aux_model['orig_feats'],name)
                handles.append(hk)
                m.register_forward_hook(hk)
        if args.dataset == 'Cifar10':
            aux_model['dcp_classifier'] = DCPClassifier(args,model)
            learnable_keys.append('dcp_classifier')
    if args.method == 'st_gRDA':

        aux_model['bn_original'] = {}
        aux_model['bn_history'] = {}
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.BatchNorm2d):
                aux_model['bn_original'][name] = m.weight.detach().clone()
                aux_model['bn_history'][name] = 0
    if args.method == "SCB":
        name_list = args.dcp_name_list
        def gethook(tarlist,name):
            def hook(model, input, output):
                tarlist[name] = output
            return hook

        aux_model['feats'] = {}
        for name,m in model.named_modules():
            if isinstance(m,custom_model.BasicBlock):
                hk = gethook(aux_model['feats'],name)
                m.register_forward_hook(hk)
                handles.append(hk)

        aux_model['scb_classifier'] = custom_model.SCBClassifier(args,model)
        learnable_keys.append('scb_classifier')

            

    if args.method == 'self_distill':
        aux_model['in_feats'] = {}
        aux_model['out_feats'] = {}
        def gethook(inlist,outlist,name):
            def hook(model, input, output):
                #print('why')
                inlist[name] = input[0]
                outlist[name] = output
            return hook
        for name,m in model.named_modules():
            if isinstance(m,BasicBlock):
                if m.conv1.in_channels == m.conv1.out_channels:
                    hk = gethook(aux_model['in_feats'],aux_model['out_feats'],name)
                    m.register_forward_hook(hk)

    if args.method == "big_kernel":
        # gather activaction map
        EPS = 1e-8
        def getacthook(tar_dict,name):
            def hook(model, input, output):
                act = torch.where(output<EPS,torch.zeros_like(output),torch.ones_like(output))
                if name in tar_dict:
                    tar_dict[name] += act
                else:
                    tar_dict[name] = act
                # may accelerate by remove if 

        aux_model['activition_map'] = {}
        aux_model['handle'] = {}
        for name,m in model.named_modules():
            if isinstance(m,torch.nn.ReLU()):
                hk = getacthook(aux_model['activition_map'],name)
                hd = m.register_forward_hook(hk)
                aux_model['handle'][name] = hd

    return model,aux_model,learnable_keys,handles

def model_to_gpu(args,model,aux_model,learnable_keys):
    ngpus_per_node = torch.cuda.device_count()
    if args.prune:
        add_prune_mask(model,args)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.method == 'st_gRDA':
                for ky in aux_model['bn_original']:
                    aux_model['bn_original'][ky].cuda(args.gpu)                
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)


        
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            for ky in aux_model:
                if isinstance(aux_model[ky],torch.nn.Module):
                    aux_model[ky] = torch.nn.parallel.DistributedDataParallel(aux_model[ky], device_ids=[args.gpu])
        else:
            model.cuda()
            for ky in aux_model:
                if isinstance(aux_model[ky], torch.nn.Module):
                    aux_model[ky].cuda()
                if isinstance(aux_model[ky],torch.Tensor):
                    aux_model[ky] = aux_model[ky].cuda()
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
            if isinstance(aux_model[ky],torch.Tensor):
                    aux_model[ky] = aux_model[ky].to(args.gpu)
        
        if args.method == 'st_gRDA':
            for ky in aux_model['bn_original']:
                aux_model['bn_original'][ky] = aux_model['bn_original'][ky].cuda(args.gpu)
                #print(ky)
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
            if isinstance(aux_model[ky],torch.Tensor):
                    aux_model[ky] = aux_model[ky].cuda()