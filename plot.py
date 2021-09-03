import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


import custom_model
plt.switch_backend('agg')
title = '139'

def add_prune_mask(model):
    print('add prune mask to model')
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            prune.l1_unstructured(module, name='weight', amount=0)
            prune.l1_unstructured(module, name='bias', amount=0)

model = custom_model.cifar10_resnet56()
model = model.cuda()
add_prune_mask(model)
model.load_state_dict(torch.load('final_{}.pth'.format(title))['state_dict'])


total = 0
for name,m in model.named_modules():
    if isinstance(m,nn.BatchNorm2d):
        dt = m.weight_orig.view(-1)
        total += dt.shape[0]

stat = torch.zeros(total)
index = 0
for name,m in model.named_modules():
    if isinstance(m,nn.BatchNorm2d):
        dt = m.weight_orig.view(-1).abs()
        #print(m.weight)
        size = dt.shape[0]
        stat[index:index+size] = dt
        index += size


stat = stat.cpu().detach().numpy()
print('num_parameters:',len(stat))
fig = plt.figure()
plt.ylim(0,300)
small = len(np.where(stat < 0.001)[0])
plt.hist(stat,100,range=[0,1])
plt.text(0.125,8000,'#para<0.001 = {}'.format(small))
plt.text(0.125,9000,'#para = {}'.format(len(stat)))
plt.savefig('distribution_margin_channel_{}'.format(title))
