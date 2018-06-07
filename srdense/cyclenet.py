import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool
from functools import partial

from .proj_utils import network as net_utils
from .proj_utils.model_utils import *

from .proj_utils.torch_utils import to_device

class MakeLayers(nn.Module):
    def __init__(self, in_channels, net_cfg, res_blocks=True, res_type='concat'):
        super(MakeLayers, self).__init__()
        pre_layers = []
        res_layers = []
        init_flag = True
        self.res_type = res_type
        self.use_residual = False
        
        if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
            for sub_cfg in net_cfg:
                sub_layer   = MakeLayers(in_channels, sub_cfg)
                in_channels = sub_layer.out_chan
                pre_layers.append(sub_layer)
        else:
            org_in_channels = in_channels

            for idx, item in enumerate(net_cfg):
                
                if item == 'M':
                    #pre_layers.append(padConv2d(in_channels,  in_channels, kernel_size = 3, stride=2, bias=False) )
                    #pre_layers.append(nn.LeakyReLU(0.1, inplace=True) )
                    pre_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    if len(item) == 2:
                        out_channels, ksize = item
                        dilation = 1
                        num_groups = 0
                    elif len(item) == 3:
                        out_channels, ksize, dilation = item
                        num_groups = 0 
                    else:
                        out_channels, ksize, dilation, num_groups = item

                    use_norm = num_groups!=0

                    if init_flag or (not res_blocks):
                        if self.res_type is 'concat':
                            pre_layers.append(passthrough()) 
                            in_channels = in_channels
                        else:
                            pre_layers.append(net_utils.pre_Conv2d_GroupNorm(in_channels, out_channels, ksize, 
                                              use_norm=use_norm, num_groups=num_groups, dilation = dilation))
                            in_channels = out_channels

                        if len(net_cfg) == 1:  
                            pre_layers.append(net_utils.pre_Conv2d_GroupNorm(in_channels, out_channels, ksize, 
                                              use_norm=use_norm, num_groups=num_groups, dilation = dilation))
                            in_channels = out_channels
                        
                        else:
                            # the following is very much needed.
                            res_layers.append(net_utils.pre_Conv2d_GroupNorm(in_channels, out_channels, ksize, 
                                            use_norm=use_norm, num_groups=num_groups, dilation = dilation))
                            init_flag = False
                            in_channels = out_channels

                    else:
                        self.use_residual = True
                        #use_relu = idx != (len(net_cfg) - 1)
                        res_layers.append(net_utils.pre_Conv2d_GroupNorm(in_channels, out_channels, ksize, 
                                          use_norm=use_norm, num_groups=num_groups, dilation = dilation))
                        in_channels = out_channels
                            
            if self.use_residual and self.res_type is 'concat':
                in_channels = out_channels + org_in_channels
            else:
                in_channels = out_channels             

        self.pre_res  =   nn.Sequential(*pre_layers)   
        self.res_path =   nn.Sequential(*res_layers)    
        self.out_chan =   in_channels
        #self.activ    =   nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs):
        # TODO do we need to add activation? 
        # CycleGan regards this. I guess to prevent spase gradients
        #import pdb; pdb.set_trace()
        pre_res = self.pre_res(inputs.contiguous())
        
        if self.use_residual:
            res_path = self.res_path(pre_res)

            if self.res_type is 'concat':
                return torch.cat([pre_res, res_path], dim=1 )
            else:
                return pre_res + res_path
        else:
            return pre_res

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        bconv = net_utils.pre_Conv2d_GroupNorm
        self.conv1   =  bconv(3,  13,  3, dilation = 1,  num_groups = None, use_norm=False)
        self.conv2   =  bconv(16, 16, 3,  dilation = 2,  num_groups = 4, use_norm=True)
        self.conv3   =  bconv(32, 16, 3,  dilation = 4,  num_groups = 4, use_norm=True)
        self.conv4   =  bconv(48, 16, 3,  dilation = 2,  num_groups = 6, use_norm=True)
        self.final   =  bconv(64, 3,  3,  dilation = 1,  num_groups = 8, use_norm=True)
        
        print(self)
    def forward(self, x):    
        conv1 = self.conv1(x)
        cout1_dense = torch.cat([x, conv1], 1)

        conv2 = self.conv2(cout1_dense)
        cout2_dense = torch.cat([x, conv1, conv2], 1)

        conv3 = self.conv3(cout2_dense)
        cout3_dense = torch.cat([x, conv1, conv2, conv3], 1)

        conv4 = self.conv4(cout3_dense)
        cout4_dense = torch.cat([x, conv1, conv2, conv3, conv4], 1)

        out = F.tanh(self.final(cout4_dense) )
        return out

class giantnet(nn.Module):
    def __init__(self):
        super(giantnet, self).__init__()
        bconv = net_utils.pre_Conv2d_GroupNorm
        
        self.conv1   =  bconv(3,  15,  3,  dilation = 1, num_groups = None, use_norm=False)
        self.conv2   =  bconv(18, 18, 3,   dilation = 2,  num_groups = 3, use_norm=True)
        self.conv3   =  bconv(36, 18, 3,   dilation = 4,  num_groups = 6, use_norm=True)
        self.conv4   =  bconv(54, 18, 3,   dilation = 4,  num_groups = 9, use_norm=True)
        self.conv5   =  bconv(72, 18, 3,   dilation = 8,  num_groups = 9, use_norm=True)
        self.conv6   =  bconv(90, 18, 3,   dilation = 2,  num_groups = 9, use_norm=True)
        self.conv7   =  bconv(108, 18, 3,  dilation = 2,  num_groups = 12, use_norm=True)
        self.conv8   =  bconv(126, 18, 3,  dilation = 2,  num_groups = 14, use_norm=True)
        self.conv9   =  bconv(144, 18, 3,  dilation = 1, num_groups = 12, use_norm=True)
        self.conv10  =  bconv(162, 18, 3,  dilation = 1, num_groups = 18, use_norm=True)
        self.conv11  =  bconv(180, 18, 3,  dilation = 1, num_groups = 20, use_norm=True)
        self.conv12  =  bconv(198, 18, 3,  dilation = 1, num_groups = 18, use_norm=True)
        
        self.final   =  bconv(216, 3, 3,   dilation = 1, num_groups = 12, use_norm=True)

        print(self)
    def forward(self, x):    
        conv1 = self.conv1(x)
        cout1_dense = torch.cat([x, conv1], 1)

        conv2 = self.conv2(cout1_dense)
        cout2_dense = torch.cat([x, conv1, conv2], 1)

        conv3 = self.conv3(cout2_dense)
        cout3_dense = torch.cat([x, conv1, conv2, conv3], 1)

        conv4 = self.conv4(cout3_dense)
        cout4_dense = torch.cat([x, conv1, conv2, conv3, conv4], 1)

        conv5 = self.conv5(cout4_dense)
        cout5_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5], 1)

        conv6 = self.conv6(cout5_dense)
        cout6_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6], 1)
        
        conv7 = self.conv7(cout6_dense)
        cout7_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1)

        conv8 = self.conv8(cout7_dense)
        cout8_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1)

        conv9 = self.conv9(cout8_dense)
        cout9_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9], 1)

        conv10 = self.conv10(cout9_dense)
        cout10_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10], 1)

        conv11 = self.conv11(cout10_dense)
        cout11_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11], 1)

        conv12 = self.conv12(cout11_dense)
        cout12_dense = torch.cat([x, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12], 1)

        out = F.tanh(self.final(cout12_dense) )
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        net_cfgs = [
            [(16, 3, 1, 0), (16, 1, 1, 4), (16, 3, 1, 4)],
            [(16, 1, 1, 0)],
            ['M', (16, 3, 1, 4), (16, 1, 1, 4), (16, 3, 1, 4)],
            ['M', (32, 3,1, 8), (32, 1, 1, 4), (32, 1, 1, 4), (32, 3, 1, 4)], 
            ['M', (32, 3, 1, 8), (32, 1, 1, 4), (32, 1, 1, 4), (32, 3, 1, 4)],
            [(64, 3, 1, 8), (32, 1, 1, 8), (32, 1, 1, 8), (64, 3, 1, 8)], 
            [(64, 1, 1, 0)],
            ['M', (64, 3, 1, 8), (32, 1, 1, 8), (32, 1, 1, 8), (64, 3, 1, 8)], 
        ]
        #import pdb; pdb.set_trace()
        self.conv_0   =  MakeLayers(3,  net_cfgs)
        self.conv_1   =  net_utils.pre_Conv2d_GroupNorm(self.conv_0.out_chan, 
                          1, 1, 1, num_groups = 8, use_norm=True)
        
        print(self)
    def forward(self, x):    
        out = self.conv_0(x)
        out = self.conv_1(out)
        return out

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff  = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss  = torch.sum(error.mean()) 
        return loss 