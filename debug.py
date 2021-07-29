
import torchvision
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import json

md = nn.Conv2d(2, 4, 3)
print(md.weight.shape)