
import torchvision
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import json
m1 = nn.Conv2d(3,4,1)
m2 = nn.Conv2d(4,5,1)
mm = nn.Sequential(m1,m2)
print(mm[0])