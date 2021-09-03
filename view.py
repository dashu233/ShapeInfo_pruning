from main import args
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from model_builder import build_model,model_to_gpu
from utils import channel_remaining

model,aux_model,learnable_keys = build_model(args)
model_to_gpu(args,model,aux_model,learnable_keys)

args.resume = 'output/cifar10_st_gRDA_pretrained/ep199.pth.tar'

if args.gpu is None:
    checkpoint = torch.load(args.resume)
else:
                # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(args.resume, map_location=loc)

model.load_state_dict(checkpoint['state_dict'])

channel_remaining(model)
    

