import torch
import torch.nn as nn



def network_slimming_loss(model,images,targets,criterion,args):
    output = model(images)
    loss = criterion(output, targets)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            loss = loss + m.weight.data.abs().sum() * args.slimming_penalty
    return loss,output

def DCP_loss(model,images,targets,criterion,stage_name,aux_model,args,tp='finetune_stage'):
    if tp == 'finetune_stage':
        assert 'now_feats' in aux_model
        output_f = model(images)
        if stage_name == 'final':
            return criterion(output_f, targets), output_f
        else:
            feat_now = aux_model['now_feats'][stage_name]
            Lf = criterion(output_f,targets)
            Lp, output_p = aux_model['dcp_classifier'](feat_now)
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
            loss, output_p = aux_model['dcp_classifier'](feat_now)
            loss = reloss + args.dcp_penalty * loss
            return loss, output_p
