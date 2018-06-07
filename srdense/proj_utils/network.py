import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

class GroupNorm(nn.Module):
    def __init__(self, num_groups,  num_features, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias   = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()    

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0, 'C {} and G {}'.format(C, G)

        x    = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True)
        
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

# class Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True):
#         super(Conv2d, self).__init__()
#         #padding = int((kernel_size - 1) / 2) if same_padding else 0
#         #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
#         self.conv = padConv2d(in_channels, out_channels, kernel_size, stride, padding=None, bias=True)

#         self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 relu=True, same_padding=True, dilation=1):
        super(Conv2d_BatchNorm, self).__init__()

        actual_ks = (kernel_size-1)*dilation + 1
        if same_padding:
            padding = int((actual_ks - 1) / 2)
        else:
            padding = 0
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False, dilation =dilation)
        self.bn   = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class pre_Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 relu=True, same_padding=True, bias=False, dilation=1):
        super(pre_Conv2d_BatchNorm, self).__init__()
        #padding = int((kernel_size - 1) / 2) if same_padding else 0
        actual_ks = (kernel_size-1)*dilation + 1
        if same_padding:
            padding = int((actual_ks - 1) / 2)
        else:
            padding = 0

        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation =dilation)
        
        
    def forward(self, x):
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = self.conv(x)
        return x

class pre_Conv2d_GroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=False, dilation=1, num_groups = 4, use_norm=True):
        super(pre_Conv2d_GroupNorm, self).__init__()
        self.use_norm = use_norm
        actual_ks = (kernel_size-1)*dilation + 1
        padding = int((actual_ks - 1) / 2)

        if self.use_norm:
            self.norm_layer = GroupNorm(num_groups, in_channels) 
        
        self.relu = nn.LeakyReLU(0.1, inplace=False) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation =dilation)
        
    def forward(self, x):
        x = self.norm_layer(x) if self.use_norm else x
        x = self.relu(x)
        x = self.conv(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda()
    return v


def variable_to_np_tf(x):
    return x.data.cpu().numpy().transpose([0, 2, 3, 1])


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)
