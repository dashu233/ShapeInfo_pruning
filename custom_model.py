
import sys
from torch._C import dtype
from torch.functional import Tensor
import torch.nn as nn
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

'''

To define a custom model, please create a function as:

def YOUR_MODEL_NAME(args, pretrain=False)
    ......
    return your_model_instance

args passed from main.py, so you can add some model configs in args.
pretrain is a boolean value. 
You should define where to get your model pretrained parameters in this function and load it to the returned model

'''


cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet44-2a3cabcb.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet56-187c023a.pt',
}

cifar100_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt',
    'resnet32': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet32-84213ce6.pt',
    'resnet44': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet44-ffe32858.pt',
    'resnet56': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet56-f2eff4c8.pt',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
    def compute_BN_mask_FLOPS(self,x):
        # x shape:1,in_channel,W,H
        # conv1 & bn1
        FLOPS = 0
        identity = x
        remain_channel = torch.sum(self.bn1.weight_mask)
        FLOPS += (self.conv1.in_channels*remain_channel*9 + remain_channel)*x.shape[2]*x.shape[3]/self.conv1.stride[0]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        remain_channel = torch.sum(self.bn2.weight_mask)
        FLOPS += (self.conv2.in_channels*remain_channel*9 + remain_channel)*out.shape[2]*out.shape[3]/self.conv2.stride[0]

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            remain_channel = torch.sum(self.downsample[1].weight_mask)
            FLOPS += (self.downsample[0].in_channels*remain_channel + remain_channel)*x.shape[2]*x.shape[3]/self.downsample[0].stride[0]

        out += identity
        out = self.relu(out)
        return out,FLOPS





class CifarResNet(nn.Module):

    def __init__(self, block, layers, args,num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #[16,16,32,64]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def compute_BN_mask_FLOPS(self,x):
        FLOPS = 0
        remain_channel = torch.sum(self.bn1.weight_mask)
        FLOPS += (self.conv1.in_channels*remain_channel*9 + remain_channel)*x.shape[2]*x.shape[3]/self.conv1.stride[0]
        x = self.conv1(x)
        x = self.bn1(x)
        for bk in self.layer1:
            x,tp = bk.compute_BN_mask_FLOPS(x)
            FLOPS += tp
        for bk in self.layer2:
            x,tp = bk.compute_BN_mask_FLOPS(x)
            FLOPS += tp
        for bk in self.layer3:
            x,tp = bk.compute_BN_mask_FLOPS(x)
            FLOPS += tp
        
        FLOPS += self.fc.in_features*self.fc.out_features
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,FLOPS


def _resnet(
    arch: str,
    layers: List[int],
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def formal(st:str):
    return st.replace('.','_')

class DCPClassifier(nn.Module):
    def __init__(self,args,model:nn.Module,losser='cross_entropy',num_class=10):
        super().__init__()
        channel_list = {}
        name_list = args.dcp_name_list
        self.discriminal_layer_list = []
        for name,m in model.named_modules():
            if name in name_list:
                name = formal(name)
                assert hasattr(m,'out_channels'),'model in name list is not a Linear or Conv2d'
                channel_list[name] = m.out_channels
                self.discriminal_layer_list.append(name)
        print('channel_list:',channel_list)
        self.dcp_loss_penalty = args.dcp_loss_penalty
        if losser == 'cross_entropy':
            self.losser = nn.CrossEntropyLoss()
        bn_layers = {}
        avg_layers = {}
        classifers = {}
        for name in name_list:
            name = formal(name)
            print(name)
            bn_layers[name] = nn.BatchNorm2d(channel_list[name])
            avg_layers[name] = nn.AdaptiveAvgPool2d((1,1))
            classifers[name] = nn.Linear(channel_list[name],num_class)
        self.bn_layers = nn.ModuleDict(bn_layers)
        self.avg_layers = nn.ModuleDict(avg_layers)
        self.classifer_dict = nn.ModuleDict(classifers)
    def forward(self,feats,y=None,name=None):
        #print(y,name)
        name = formal(name)
        assert name in self.discriminal_layer_list,name
        bi = self.bn_layers[name](feats)
        oi = self.avg_layers[name](bi)
        oi = oi.view(oi.shape[0], -1)
        pi = self.classifer_dict[name](oi)
        loss = self.losser(pi, y)
        return loss,pi


class SCBClassifier(nn.Module):
    def __init__(self,args,model:nn.Module,losser='cross_entropy',num_class=10):
        super().__init__()
        self.scb_list = []
        self.scb_channel_list = {}
        for name,m in model.named_modules():
            name = formal(name)
            if isinstance(m,BasicBlock):
                self.scb_list.append(name)
                self.scb_channel_list[name] = m.conv2.out_channels
        
        self.scb_loss_penalty = args.scb_loss_penalty
        if losser == 'cross_entropy':
            self.losser = nn.CrossEntropyLoss()
        
        conv_layers = {}
        bn_layers = {}
        avg_layers = {}
        classifers = {}
        self.relu = nn.ReLU(inplace=True)
        for name in self.scb_list:
            name = formal(name)
            #print(name)
            conv_layers[name] = nn.Conv2d(self.scb_channel_list[name],self.scb_channel_list[name],
                        kernel_size=3,padding=1,bias=False)
            bn_layers[name] = nn.BatchNorm2d(self.scb_channel_list[name])
            avg_layers[name] = nn.AdaptiveAvgPool2d((1,1))
            classifers[name] = nn.Linear(self.scb_channel_list[name],num_class)
        self.bn_layers = nn.ModuleDict(bn_layers)
        self.conv_layers = nn.ModuleDict(conv_layers)
        self.avg_layers = nn.ModuleDict(avg_layers)
        self.classifer_dict = nn.ModuleDict(classifers)
    def forward(self,feats,y):
        #print(y,name)
        loss = 0
        for name in feats:
            feat = feats[name]
            name = formal(name)
            ci = self.relu(self.conv_layers[name](feat))
            bi = self.bn_layers[name](ci)
            oi = self.avg_layers[name](bi)
            oi = oi.view(oi.shape[0], -1)
            pi = self.classifer_dict[name](oi)
            loss += self.losser(pi, y)
        return loss * self.scb_loss_penalty


def cifar10_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar10_resnet56(*args, **kwargs) -> CifarResNet: pass


def cifar100_resnet20(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet32(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet44(*args, **kwargs) -> CifarResNet: pass
def cifar100_resnet56(*args, **kwargs) -> CifarResNet: pass



thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for layers, model_name in zip([[3]*3, [5]*3, [7]*3, [9]*3],
                                  ["resnet20", "resnet32", "resnet44", "resnet56"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_resnet,
                    arch=model_name,
                    layers=layers,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )


def cl_conv(orig:nn.Conv2d,mpfc,last_mpfc):
    if last_mpfc is None:
        tmp = nn.Conv2d(orig.in_channels, mpfc.shape[0], 
            kernel_size=orig.kernel_size, padding=orig.padding,stride=orig.stride)
        tmp.weight = orig.weight[mpfc,:,:,:].clone()
        return tmp
    else:
        tmp = nn.Conv2d(last_mpfc.shape[0], mpfc.shape[0], 
            kernel_size=orig.kernel_size, padding=orig.padding,stride=orig.stride)
        tmp.weight = orig.weight[mpfc,last_mpfc,:,:].clone()
        return tmp



def slim_CifarResNet(md:CifarResNet):
    # conv1 and bn1
    mask = md.bn1.weight_mask
    mpfc = torch.where(mask>0)[0]
    tmp = nn.Conv2d(3, mpfc.shape[0], kernel_size=3, padding=1)
    tmp.weight = md.conv1.weight[mpfc,:,:,:]
    md.conv1 = tmp

    tmp = nn.BatchNorm2d(mpfc.shape[0])
    tmp.running_mean = md.bn1.running_mean.clone()[mpfc]
    tmp.running_var = md.bn1.running_var.clone()[mpfc]

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)