
import torchvision
import torch
import torch.nn.utils.prune as prune
import json

t11111 = torch.Tensor([2,2])
t22222 = torch.where(t11111>0,torch.ones_like(t11111),torch.zeros_like(t11111))
print(t22222.dtype)
print(t22222/8)