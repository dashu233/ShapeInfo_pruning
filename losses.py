import torch
import torch.nn as nn
from custom_model import DCPClassifier

def st_margin_loss(model,images,targets,criterion,args):
    output = model(images)
    loss = criterion(output, targets)

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
    
    tpk = torch.topk(bn,int(total*args.margin_mask_rate),largest=False)
    indice = tpk.indices
    loss += args.margin_tmp_penalty * torch.sum(bn[indice].square())
    
    # for m in model.modules():
    #     if isinstance(m,nn.BatchNorm2d):
    #         mask_rate = (1-args.next_remain_rate)*1.2
    #         tpk = torch.topk(m.weight.data.abs(),int(m.weight.data.size(0)*mask_rate),largest=False)
    #         indice = tpk.indices
    #         #print(indice)
    #         loss += args.margin_tmp_penalty * torch.sum(torch.square(m.weight.data[indice]))
            
    return loss,output

def network_slimming_loss(model,images,targets,criterion,args):
    output = model(images)
    loss = criterion(output, targets)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            loss = loss + m.weight.data.abs().sum() * args.slimming_penalty
    return loss,output

def DCP_loss(model,images,targets,criterion,stage_name,aux_model,args,tp='finetune_stage'):
    #print(type(aux_model['dcp_classifier']))
    if tp == 'finetune_stage':
        assert 'now_feats' in aux_model
        output_f = model(images)
        if stage_name == 'final':
            return criterion(output_f, targets), output_f
        else:
            feat_now = aux_model['now_feats'][stage_name]
            #print(stage_name)
            Lf = criterion(output_f,targets)
            Lp, output_p = aux_model['dcp_classifier'](feat_now,y=targets,name=stage_name)
            return Lp+Lf, output_f
    elif tp == 'criterion':
        assert 'orig_feats' in aux_model
        assert 'now_feats' in aux_model
        assert 'proto_model' in aux_model
        assert 'dcp_classifier' in aux_model
        
        proto_model = aux_model['proto_model']
        output_f = model(images)
        with torch.no_grad():
            _ = proto_model(images)
        if stage_name == 'final':
            return criterion(output_f, targets), output_f
        else:
            feat_orig = aux_model['orig_feats'][stage_name]
            feat_now = aux_model['now_feats'][stage_name]
            os = (feat_orig - feat_now).view(feat_orig.shape[0], feat_orig.shape[1], -1)
            reloss = torch.mean(torch.norm(os, dim=2)) / os.shape[-1] / 2.0
            loss, output_p = aux_model['dcp_classifier'](feat_now,y=targets,name=stage_name)
            loss = reloss + args.dcp_loss_penalty * loss
            return loss, output_p
