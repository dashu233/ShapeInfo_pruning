import torchvision
import torch

md = torch.nn.Conv2d(2,4,3)
loss = torch.sum(md.weight)
loss.backward()
print(md.weight.grad.shape)