import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from utils import is_main_process,accuracy,adjust_learning_rate,AverageMeter,ProgressMeter
from losses import  network_slimming_loss,DCP_loss
import time
import random
import os

def remain_rate(epoch,args):
    if args.method == 'DCP':
        pass
    else:
        assert epoch in args.prune_steps
        ids = args.prune_steps.index(epoch)
        return (1-args.prune_rate)**((ids+1)/len(args.prune_steps))

def prune_model(train_loader,train_sampler,val_loader,model,
        criterion,optimizer,args,writer,aux_model):
    best_acc1 = 0
    remain_percent = 1
    ngpus_per_node = torch.cuda.device_count()
    if not args.prune:
        return -1
    if args.method == 'network_slimming':
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
                        index += size
                        bn[index:(index + size)] = m.weight.data.abs().clone()

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
        stage_layers_name = args.stage_layers_name
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
            st = 0 if ind == 0 else (all_layers_name.index(stage[ind - 1])+1)
            end = all_layers_name.index(stage_layers_name[ind])+1
            prune_list = all_layers_name[st:end]

            # compute criterion
            for name,m in model.named_modules():
                if name in prune_list and isinstance(m,nn.Conv2d):
                    out_channels = m.out_channels
                    prune_channel = min(int(args.prune_rate*out_channels),out_channels-1)

                    for _ in range(prune_channel):
                        if args.distributed:
                            train_sampler.set_epoch(random.randint(0, 1000))
                        for it, images, target in enumerate(train_loader):
                            if args.gpu is not None:
                                images = images.cuda(args.gpu, non_blocking=True)
                            if torch.cuda.is_available():
                                target = target.cuda(args.gpu, non_blocking=True)
                            loss, output = DCP_loss(model, images, target, criterion, aux_para['stage'], aux_model,
                                                    args,tp='criterion')
                            loss.backward()
                            if it > args.dcp_prune_criterion_iter:
                                break
                        gd = m.weight.data.grad
                        mask = m.weight_mask.detach().clone()
                        crt = torch.norm(gd.view(gd.shape[0],-1),dim=1)
                        remain_channel = torch.where(mask[:,0,0,0]>0)[0]
                        pid = torch.argmin(crt[remain_channel])
                        pc = remain_channel[pid]
                        mask[pc,:,:,:]=0
                        prune.custom_from_mask(m, 'weight', mask)

                        optimizer.zero_grad()
                        for ep in range(len(args.dcp_criterion_finetune_epoch)):
                            for it, images, target in enumerate(train_loader):
                                if args.gpu is not None:
                                    images = images.cuda(args.gpu, non_blocking=True)
                                if torch.cuda.is_available():
                                    target = target.cuda(args.gpu, non_blocking=True)
                                loss, output = DCP_loss(model, images, target, criterion, aux_para['stage'], aux_model,
                                                    args, tp='criterion')
                                loss.backward()
                                optimizer.step()




def compute_global_step(it,epoch,train_loader_length,args,aux_para):
    if args.method == 'DCP':
        assert 'stage' in aux_para
        ind = args.stage_layers_name.index(aux_para['stage'])
        if it < 0:
            # iter < 0 means only consider epoch
            return epoch + args.epoch_per_stage * ind
        else:
            return it + (epoch + args.epoch_per_stage * ind) * train_loader_length
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
    if args.method == 'DCP':
        pass
    else:
        raise NotImplementedError

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        if args.method == 'network_slimming':
            loss,output = network_slimming_loss(model,images,target,criterion,args)
        elif args.method == 'DCP':
            assert 'stage' in aux_para
            loss,output = DCP_loss(model,images,target,criterion,aux_para['stage'],aux_model,args)
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print('loss:',loss)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                    writer.add_scalar('stage_index', args.stage_layers_name.index(aux_para['stage']), step)


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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
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
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
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
    return Acc_orig.avg, Acc_v.avg, Acc_h.avg, Diff_v.avg, Diff_h.avg


