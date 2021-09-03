import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn as nn
from utils import is_main_process,accuracy,adjust_learning_rate,AverageMeter,ProgressMeter


from utils import is_main_process,accuracy,adjust_learning_rate,AverageMeter,ProgressMeter,channel_remaining
from losses import  network_slimming_loss,DCP_loss,SCB_loss
import time
import random
import os

def cal_activition_rate(aux_model,times):
    assert 'activition_map' in aux_model
    act_map = aux_model['activition_map']
    for ky in act_map:
        act_map = act_map[ky]/times

def remove_hooks(aux_model):
    assert 'handle' in aux_model
    hds = aux_model['handle']
    for ky in hds:
        hds[ky].remove()
    

def remain_rate(epoch,args):
    if args.method == 'DCP':
        pass
    else:
        assert epoch in args.prune_steps
        ids = args.prune_steps.index(epoch)
        return (1-args.prune_rate)**((ids+1)/len(args.prune_steps))

def next_remain_rate(epoch,args):
    for ep in args.prune_steps:
        if ep > epoch:
            break
    ids = args.prune_steps.index(ep)
    return (1-args.prune_rate)**((ids+1)/len(args.prune_steps))

def prune_model(train_loader,train_sampler,val_loader,model,
        criterion,optimizer,args,writer,aux_model):
    best_acc1 = 0
    remain_percent = 1
    ngpus_per_node = torch.cuda.device_count()
    if not args.prune:
        return -1

    if args.method == 'SCB':
        for epoch in range(args.start_epoch, args.epochs):
            #args.margin_tmp_penalty = min(epoch,args.margin_warm_epoch)/args.margin_warm_epocg * args.margin_penalty
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args, writer)
            train(train_loader, model, criterion, optimizer, epoch, args, writer, aux_model)
            acc1 = validate(val_loader, model, criterion, args, epoch, writer, aux_model)
            _ = flip_validate(val_loader, model, criterion, args, epoch, writer)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if writer is not None:
                writer.add_scalar('remain_percent', remain_percent, epoch)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if epoch % args.checkpoint_interval == args.checkpoint_interval - 1:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'ep{:0>3}.pth.tar'.format(epoch)))
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'best_model.pth.tar'))
    if args.method == 'self_distill':
        for epoch in range(args.start_epoch, args.epochs):
            
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args, writer)
            train(train_loader, model, criterion, optimizer, epoch, args, writer, aux_model)
            acc1 = validate(val_loader, model, criterion, args, epoch, writer, aux_model)
            _ = flip_validate(val_loader, model, criterion, args, epoch, writer)

            if epoch in args.prune_steps:
                remain_percent = remain_rate(epoch, args)
                
                total = 0
                for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        total += m.weight.data.shape[0]

                bn = torch.zeros(total)
                index = 0
                for name,m in model.named_modules():
                    if isinstance(m, nn.Conv2d):
                        size = m.weight.data.shape[0]
                        bn[index:(index + size)] = m.weight.data.abs().clone()
                        index += size

                y, i = torch.sort(bn, descending=True)
                thre_index = int(total * remain_percent)
                thre = y[thre_index]

                pruned = 0
                for k, m in enumerate(model.modules()):
                    if isinstance(m, nn.BatchNorm2d):
                        weight_copy = m.weight.data.clone()
                        mask = weight_copy.abs().gt(thre).float().cuda()
                        pruned = pruned + mask.shape[0] - torch.sum(mask)
                        prune.custom_from_mask(m, 'weight', mask)
                        prune.custom_from_mask(m, 'bias', mask)
                        
                if is_main_process(args):
                    channel_remaining(model)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if writer is not None:
                writer.add_scalar('remain_percent', remain_percent, epoch)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if epoch % args.checkpoint_interval == args.checkpoint_interval - 1:
                    for ky in model.state_dict():
                        print(ky)
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'ep{:0>3}.pth.tar'.format(epoch)))
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'best_model.pth.tar'))

    if args.method == 'network_slimming' or args.method == 'st_gRDA' :
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args, writer)
            train(train_loader, model, criterion, optimizer, epoch, args, writer, aux_model)
            acc1 = validate(val_loader, model, criterion, args, epoch, writer, aux_model)
            _ = flip_validate(val_loader, model, criterion, args, epoch, writer)

            if epoch in args.prune_steps:
                remain_percent = remain_rate(epoch, args)
                total = 0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        total += m.weight.data.shape[0]
                bn = torch.zeros(total)
                index = 0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        size = m.weight.data.shape[0]
                        bn[index:(index + size)] = m.weight.data.abs().clone()
                        index += size

                y, i = torch.sort(bn, descending=True)
                thre_index = int(total * remain_percent)
                thre = y[thre_index]

                pruned = 0
                for k, m in enumerate(model.modules()):
                    if isinstance(m, nn.Conv2d):
                        size = m.weight.data.shape[0]
                        weight_copy = torch.norm(m.weight.data.view(size,-1),dim=1).clone()
                        mask = weight_copy.abs().gt(thre).float().cuda()
                        pruned = pruned + mask.shape[0] - torch.sum(mask)
                        full_mask = mask.repeat(m.in_channels,m.kernel_size[0],m.kernel_size[1],1)
                        full_mask = full_mask.permute(3,0,1,2)
                        prune.custom_from_mask(m, 'weight', full_mask)
                        if is_main_process(args):
                            print('layer index: {:d} \t total channel: {} \t remaining channel: {:d}'.
                                  format(k, m.weight.shape[0], int(torch.sum(m.weight_mask[:,0,0,0]))))
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if writer is not None:
                writer.add_scalar('remain_percent', remain_percent, epoch)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if epoch % args.checkpoint_interval == args.checkpoint_interval - 1:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'ep{:0>3}.pth.tar'.format(epoch)))
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'best_model.pth.tar'))

    if args.method == 'network_slimming' or args.method == 'st_gRDA' or args.method == 'st_margin':
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            args.margin_tmp_penalty = min(epoch+1,args.margin_warm_epoch)/args.margin_warm_epoch * args.margin_penalty
            args.next_remain_rate = next_remain_rate(epoch,args)
            print('next rate:',args.next_remain_rate)
            adjust_learning_rate(optimizer, epoch, args, writer)
            train(train_loader, model, criterion, optimizer, epoch, args, writer, aux_model)
            acc1 = validate(val_loader, model, criterion, args, epoch, writer, aux_model)
            _ = flip_validate(val_loader, model, criterion, args, epoch, writer)

            if (not args.prune_last and (epoch in args.prune_steps)) or (args.prune_last and epoch == args.prune_steps[-1]):
                remain_percent = remain_rate(epoch, args)
                args.remain_percent = remain_percent
                if args.use_global_criterion:
                    print('using global criterion')
                    total = 0
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            total += m.weight.data.shape[0]
                    bn = torch.zeros(total)
                    index = 0
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            size = m.weight.data.shape[0]
                            if args.scaling_norm_criterion:
                                bn[index:(index + size)] = m.weight.data.abs().clone()/torch.sum(m.weight.data.abs())
                            else:
                                bn[index:(index + size)] = m.weight.data.abs().clone()
                            index += size

                    y, i = torch.sort(bn, descending=True)
                    thre_index = int(total * remain_percent)
                    thre = y[thre_index]

                    pruned = 0
                    for k, m in enumerate(model.modules()):
                        if isinstance(m, nn.BatchNorm2d):
                            weight_copy = m.weight.data.clone()
                            if args.scaling_norm_criterion:
                                mask = (weight_copy.abs()/torch.sum(m.weight.data.abs())).gt(thre).float().cuda()
                            else:
                                mask = weight_copy.abs().gt(thre).float().cuda()
                            pruned = pruned + mask.shape[0] - torch.sum(mask)
                            prune.custom_from_mask(m, 'weight', mask)
                            prune.custom_from_mask(m, 'bias', mask)
                            if is_main_process(args):
                                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                                    format(k, int(m.weight.shape[0]), int(sum(m.weight_mask))))
                else:
                    print('using local criterion')
                    pruned = 0
                    for k, m in enumerate(model.modules()):
                        if isinstance(m, nn.BatchNorm2d):
                            if args.scaling_norm_criterion:
                                weight_copy=m.weight.data.abs().clone()/torch.sum(m.weight.data.abs())
                            else:
                                weight_copy = m.weight.data.abs().clone()
                            mask = torch.zeros_like(weight_copy)
                            _,indice = torch.topk(weight_copy,int(weight_copy.size(0)*remain_percent),largest=True)
                            mask[indice] = 1
                            pruned = pruned + mask.shape[0] - torch.sum(mask)
                            prune.custom_from_mask(m, 'weight', mask)
                            prune.custom_from_mask(m, 'bias', mask)
                            if is_main_process(args):
                                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                                    format(k, int(m.weight.shape[0]), int(sum(m.weight_mask))))

                
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if writer is not None:
                writer.add_scalar('remain_percent', remain_percent, epoch)
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if epoch % args.checkpoint_interval == args.checkpoint_interval - 1:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'ep{:0>3}.pth.tar'.format(epoch)))
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join('output', args.expname, 'best_model.pth.tar'))


    elif args.method == 'DCP':
        all_layers_name = args.all_layer_names
        stage_layers_name = args.dcp_name_list
        assert 'proto_model' in aux_model
        assert 'dcp_classifier' in aux_model
        assert 'orig_feats' in aux_model
        assert 'now_feats' in aux_model

        for ind,stage in enumerate(stage_layers_name):
            aux_para = {'stage':stage}
            for epoch in range(args.dcp_epoch_per_stage):
                if args.distributed:
                    train_sampler.set_epoch(epoch)
                train(train_loader, model, criterion, optimizer, epoch, args, writer, aux_model, aux_para)
            st = 0 if ind == 0 else (all_layers_name.index(stage_layers_name[ind - 1])+1)
            end = all_layers_name.index(stage_layers_name[ind])+1
            prune_list = all_layers_name[st:end]
            print('all layer to prune in stage {}:{}'.format(stage,prune_list))
            # compute criterion
            for name,m in model.named_modules():
                #print(name)
                if name in prune_list and isinstance(m,nn.Conv2d):
                    out_channels = m.out_channels
                    prune_channel = min(int(args.prune_rate*out_channels),out_channels-1)

                    for _ in range(prune_channel):
                        if args.distributed:
                            train_sampler.set_epoch(random.randint(0, 1000))
                        for it, (images, target) in enumerate(train_loader):
                            if args.gpu is not None:
                                images = images.cuda(args.gpu, non_blocking=True)
                                target = target.cuda(args.gpu, non_blocking=True)
                            if torch.cuda.is_available():
                                images = images.cuda(non_blocking=True)
                                target = target.cuda(non_blocking=True)
                            loss, output = DCP_loss(model, images, target, criterion, stage, aux_model,
                                                    args,tp='criterion')
                            loss.backward()
                            
                            if it > args.dcp_prune_criterion_iter:
                                break
                        gd = m.weight_orig.grad
                        mask = m.weight_mask.detach().clone()
                        crt = torch.norm(gd.view(gd.shape[0],-1),dim=1)
                        remain_channel = torch.where(mask[:,0,0,0]>0)[0]
                        pid = torch.argmin(crt[remain_channel])
                        pc = remain_channel[pid]
                        mask[pc,:,:,:]=0
                        prune.custom_from_mask(m, 'weight', mask)

                        optimizer.zero_grad()
                        for ep in range(args.dcp_criterion_finetune_epoch):
                            for it, (images, target) in enumerate(train_loader):
                                if args.gpu is not None:
                                    images = images.cuda(args.gpu, non_blocking=True)
                                    target = target.cuda(args.gpu, non_blocking=True)
                                if torch.cuda.is_available():
                                    images = images.cuda(non_blocking=True)
                                    target = target.cuda(non_blocking=True)
                                loss, output = DCP_loss(model, images, target, criterion,stage, aux_model,
                                                    args, tp='criterion')
                                loss.backward()
                                optimizer.step()




def compute_global_step(it,epoch,train_loader_length,args,aux_para):
    if args.method == 'DCP':
        assert 'stage' in aux_para
        ind = args.dcp_name_list.index(aux_para['stage'])
        if it < 0:
            # iter < 0 means only consider epoch
            return epoch + args.dcp_epoch_per_stage * ind
        else:
            return it + (epoch + args.dcp_epoch_per_stage * ind) * train_loader_length
    else:
        if it < 0:
            # iter < 0 means only consider epoch
            return epoch
        else:
            return it + epoch * train_loader_length

def train(train_loader, model, criterion, optimizer, epoch, args, writer=None,
          aux_model=None,aux_para=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))


    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        elif torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if args.method == 'DCP':
            assert 'stage' in aux_para
            loss,output = DCP_loss(model,images,target,criterion,aux_para['stage'],aux_model,args)
        elif args.method == 'self_distill':
            #print('aux:',aux_para)
            assert 'in_feats','out_feats' in aux_model
            output = model(images)
            loss = criterion(output,target)
            loss_p = 0
            for name in aux_model['in_feats']:
                #print('in_feats',type(aux_model['in_feats'][name]))
                #print('out_feats',type(aux_model['out_feats'][name]))
                loss_p += torch.norm((aux_model['in_feats'][name]-aux_model['out_feats'][name]).view(output.shape[0],-1),dim=1).mean()
            loss += loss_p * args.distill_penalty
        elif args.method == 'SCB':
            loss,output = SCB_loss(model,images,target,criterion,aux_model,args)
        else:
            output = model(images)
            loss = criterion(output,target)
            # compute gradient and do SGD step
            # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.method == 'st_gRDA':
            assert 'bn_history' in aux_model
            assert 'bn_original' in aux_model
            for name,m in model.named_modules():
                if name in aux_model['bn_history']:
                    if m.weight.data.grad is not None:
                        aux_model['bn_history'][name] = aux_model['bn_history'][name] + m.weight.data.grad
                    else:
                        print(name)
        elif args.method == 'network_slimming':
            for name,m in model.named_modules():
                if isinstance(m,nn.BatchNorm2d):
                    m.weight_orig.grad += torch.where(m.weight_orig>0,1.0,-1.0)*args.slimming_penalty
        elif args.method == 'st_margin':
            total = 0
            for m in model.modules():
                if isinstance(m,nn.BatchNorm2d):
                    total += m.weight.data.size(0)
            
            bn = torch.zeros(total).cuda()
            index = 0

            for m in model.modules():
                if isinstance(m,nn.BatchNorm2d):
                    size = m.weight.data.size(0)
                    bn[index:index+size] = m.weight.data.abs()
                    index += size
            
            tpk = torch.topk(bn,int(total*(1-args.next_remain_rate)*1.01),largest=False)
            thr = torch.max(tpk.values)
            
            
            for m in model.modules():
                if isinstance(m,nn.BatchNorm2d):
                    tmp = torch.where(m.weight>thr,torch.zeros_like(m.weight),m.weight.detach().clone()).detach().clone()
                    m.weight_orig.grad += args.margin_tmp_penalty * tmp
                    #print('grad_sum',torch.sum(m.weight_orig.grad))


        optimizer.step()
        if args.method == 'st_gRDA':
            assert 'bn_history' in aux_model
            assert 'bn_original' in aux_model
            for name,m in model.named_modules():
                if name in aux_model['bn_history']:
                    step = compute_global_step(i,epoch,len(train_loader),args,aux_para)
                    m.weight_orig.data = gRDA_update(aux_model['bn_original'][name],aux_model['bn_history'][name],args,step)
        

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print('loss:',loss)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if writer is not None:
                step = compute_global_step(i,epoch,len(train_loader),args,aux_para)
                writer.add_scalar('time', time.time() - end, step)
                writer.add_scalar('top1_train_iter', acc1[0], step)
                writer.add_scalar('top5_train_iter', acc5[0], step)
                writer.add_scalar('loss_train_iter', loss.item(), step)
                if args.method == 'DCP':
                    writer.add_scalar('stage_index', args.dcp_name_list.index(aux_para['stage']), step)


def validate(val_loader, model, criterion, args, epoch, writer=None,aux_para=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    if writer is not None:
        step = compute_global_step(-1,epoch,-1,args,aux_para)
        writer.add_scalar('top1_val_ep', top1.avg, step)
        writer.add_scalar('top5_val_ep', top5.avg, step)
    return top1.avg


def flip_validate(val_loader, model, criterion, args, epoch, writer=None,aux_para=None):
    batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    Acc_orig = AverageMeter('acc_orig', ':6.2f')

    Acc_h = AverageMeter('acc_h', ':6.2f')
    Acc_v = AverageMeter('acc_v', ':6.2f')
    Diff_h = AverageMeter('diff_h', ':6.2f')
    Diff_v = AverageMeter('diff_v', ':6.2f')


    progress = ProgressMeter(
        len(val_loader),
        [batch_time, Acc_orig, Acc_h, Acc_v, Diff_h, Diff_v],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
                
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            output_v = model(images.flip([2]))
            output_h = model(images.flip([3]))

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            acc_v = accuracy(output_v, target, topk=(1,))
            acc_h = accuracy(output_h, target, topk=(1,))
            Acc_orig.update(acc1[0].item(), images.size(0))
            Acc_v.update(acc_v[0].item(), images.size(0))
            Acc_h.update(acc_h[0].item(), images.size(0))
            sm = nn.Softmax(dim=1)
            so1 = sm(output)
            so_v = sm(output_v)
            so_h = sm(output_h)
            diff_v = torch.mean(torch.sum((so1 - so_v).abs(), dim=1)).item()
            diff_h = torch.mean(torch.sum((so1 - so_h).abs(), dim=1)).item()
            Diff_h.update(diff_h, images.size(0))
            Diff_v.update(diff_v, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        print('Acc:{:.3f}, Acc_h:{:.3f}, Acc_v:{:.3f},Diff_h:{:.3f}, Diff_v:{:.3f}'.format(
            Acc_orig.avg, Acc_h.avg, Acc_v.avg, Diff_h.avg, Diff_v.avg
        ))

    if writer is not None:
        step = compute_global_step(-1,epoch,-1,args,aux_para)
        writer.add_scalar('acc_val_ep', Acc_orig.avg, step)
        writer.add_scalar('acc_v_val_ep', Acc_v.avg, step)
        writer.add_scalar('acc_h_val_ep', Acc_h.avg, step)
        writer.add_scalar('diff_v_val_ep', Diff_v.avg, step)
        writer.add_scalar('diff_h_val_ep', Diff_h.avg, step)
        writer.add_scalar('diff_acc_val_ep', Acc_h.avg-Acc_orig.avg, step)
    return Acc_orig.avg, Acc_v.avg, Acc_h.avg, Diff_v.avg, Diff_h.avg

def gRDA_update(original,history,args,it):
    # print(original.device,history.device)
    thr = args.st_gRDA_c* (args.lr)**0.5 * (it*args.lr)**args.st_gRDA_mu
    tmp = original - history * args.lr
    return torch.where(tmp > thr, tmp-thr,torch.zeros_like(tmp))