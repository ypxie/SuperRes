import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .torch_utils import *
#from .local_utils import Indexflow, split_img
from collections import deque, OrderedDict
import functools

class passthrough(nn.Module):
    def __init__(self, **kwargs):
        super(passthrough, self).__init__()
    def forward(self, x, **kwargs):
        return x

class pretending_norm(nn.Module):
    def __init__(self, nchans, **kwargs):
        super(pretending_norm, self).__init__()
    def forward(self, x, **kwargs):
        return x

def resize_layer(inputs, sizes):
    dst_row, dst_col = sizes
    org_row, org_col = inputs.size()[2], inputs.size()[3]
    if dst_row == org_row and dst_col == org_col:
        return inputs

    if dst_row > org_row and dst_col > org_col:
       return F.upsample_bilinear(inputs, (dst_row, dst_col) ) 
    elif dst_row < org_row and dst_col < org_col:   
        return F.adaptive_avg_pool2d(inputs, (dst_row, dst_col) ) 
    else:
        max_row = max(dst_row, org_row)
        max_col = max(dst_col, org_col)
        max_inputs = F.upsample_bilinear(inputs, (max_row, max_col) )    
        
        return F.adaptive_avg_pool2d(max_inputs, (dst_row, dst_col))

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

class padConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, 
                 stride=1, padding=None ,bias=False, dilation=1):
        super(padConv2d, self).__init__()

        if padding is None:
            left_row  = (kernel_size - 1) //2 
            right_row = (kernel_size - 1) - left_row
            left_col  = (kernel_size - 1) //2
            right_col = (kernel_size - 1) - left_col
            self.padding = (left_row, right_row, left_col, right_col)
        elif type(padding) is int:
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding  

        self.conv2d  = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, 
                                 padding=0, bias=bias, stride=stride)

    def forward(self, inputs):
        padding = F.pad(inputs, self.padding, mode='reflect')
        output  = self.conv2d(padding)
        return output

def up_conv(in_dim, out_dim, norm, activ, repeat=1, get_layer = False):
    _layers = [nn.Upsample(scale_factor=2,mode='nearest')]
    _layers += [padConv2d(in_dim,  in_dim, kernel_size = 3, stride=stride, bias=False)]
    _layers += [getNormLayer(norm)(out_dim )]
    _layers += [activ]

    for _ in range(repeat-1):
        _layers += [padConv2d(out_dim,  out_dim,  kernel_size = 1, padding=0)]
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ]
        
    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers

def down_conv(in_dim, out_dim, norm, activ, repeat=1,
              kernel_size=3, get_layer = False):
    _layers = [padConv2d(in_dim,  out_dim, kernel_size = 3, stride=2, bias=False)]
    _layers += [getNormLayer(norm)(out_dim )]
    _layers += [activ]
    for _ in range(repeat):
        _layers += [padConv2d(out_dim,  out_dim, kernel_size = 1, padding=0, bias=False)]
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ]
    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers

def conv_norm(in_dim, out_dim, norm='bn', activ=None, repeat=1,   get_layer = False,
              last_active=True, kernel_size=1, padding=None, stride=1, last_norm=True):
    _layers = []
    _layers += [padConv2d(in_dim,  out_dim, kernel_size = kernel_size,padding=padding, stride=stride, bias=False)]

    for _ in range(repeat):
        _layers += [getNormLayer(norm)(out_dim )]
        _layers += [activ] 
        _layers += [padConv2d(out_dim,  out_dim, kernel_size = kernel_size, padding=padding, bias=False)]
        
    if last_norm:
       _layers += [getNormLayer(norm)(out_dim )]

    if last_active and activ is not None:
       _layers += [activ]
    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers

def brach_out(in_dim, out_dim, norm, activ, repeat= 1, get_layer = False):
    _layers = []
    for _ in range(repeat):
        _layers += [padConv2d(in_dim,  in_dim, kernel_size = 3, stride= 1, bias=False)]
        _layers += [getNormLayer(norm)(in_dim )]
        _layers += [activ]
    
    _layers += [padConv2d(in_dim,  out_dim, 
                kernel_size = 1, padding=0, bias=False)]    
    _layers += [nn.Tanh()]

    if get_layer:
        return nn.Sequential(*_layers)
    else:
        return _layers


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, kernel_size=3, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = padConv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = padConv2d(planes, planes, kernel_size=kernel_size, stride=1, # change
                               padding= (kernel_size-1)//2, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = padConv2d(planes, planes , kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes )
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class LayerNorm1d(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  

class LayerNorm2d(nn.Module):
    ''' 2D Layer normalization module '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.eps = eps

    def forward(self, input):
        b,c,h,w = input.size()
        x = input.view(b, -1)
        mean = x.mean(-1, keepdim=True).view(b,1,1,1)
        std = x.std(-1, keepdim=True).view(b,1,1,1)

        out = self.gamma * (input -  mean)
        factor = (std + self.eps) + self.beta  
        out = out / factor
        return out

def getNormLayer(norm='bn', dim=2):

    norm_layer = None
    if dim == 2:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm == 'ln':
            norm_layer = functools.partial(LayerNorm2d)
    elif dim == 1:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm1d, affine=False)
        elif norm == 'ln':
            norm_layer = functools.partial(LayerNorm1d)
    assert(norm_layer != None)
    return norm_layer

def batch_forward(cls, BatchData, batch_size,**kwards):
    total_num = BatchData.shape[0]
    results = []
    for ind in Indexflow(total_num, batch_size, False):
        data = BatchData[ind]
        results.append(cls.forward(data, **kwards))
    return torch.cat(results, dim=0)


def spatial_pool(x, keepdim=True):
    # input should be of N * channel * row * col
    x = torch.mean(x, -1, keepdim=False)
    x = torch.mean(x, -1, keepdim=False)
    if keepdim:
        return x.unsqueeze(-1).unsqueeze(-1)
    else:
        return x

class spatialAttention(nn.Module):
    # accept a query feat maps and memory feat maps
    # (b, len, c, row, col)
    # return memory_size output, gated by stride conv
    def __init__(self, query_chans, mem_chans, dilation_list=None, activ='selu', norm='bn'):
        super(spatialAttention, self).__init__()

        self.query_chans = query_chans
        self.mem_chans   = mem_chans
        self.conv_query  = ConvBN(query_chans, mem_chans, activ = activ, norm=norm)

        self.conv_ops = _make_nConv(query_chans + mem_chans, mem_chans, 
                                    depth=len(dilation_list[0:-1]), norm=norm, 
                                    activ = activ, dilation_list = dilation_list[0:-1] )
        self.final_conv = ConvBN(mem_chans, mem_chans, activ=activ, dilation=dilation_list[-1], 
                                 act= None, norm=norm)

    def forward(self, keys, mems):
        '''
        Parameters
        ---------
        keys: of size (b, c, row, col)
        mems: of size (b, c_mem, row, col)
        Returns:
        --------
        tensor of size (b, c_mem, row, col)
        ''' 
        inputs = torch.cat([keys, mems], dim=1)
        out_tensor = self.conv_ops(inputs)
        
        out_gate = F.sigmoid(self.final_conv(out_tensor))
        query_maps = self.conv_query(keys)
        
        output = out_gate * mems + (1-out_gate) * query_maps
        return output

def temperal_atten(querys, memory, strengths, atten_mask=None):
    # queries (batch_size, que_slot, query_size)
    # memory  (batch_size, mem_slot, query_size)
    # strength (bs, que_slot, 1)
    # return  (batch_size, query_size, mem_slot)
    distance = cosine_distance(memory, torch.transpose(querys,2,1).contiguous())
    #print('strengths size: ', strengths.size(), distance.size())
    strengths = torch.transpose(strengths, 1, 2).expand_as(distance)
    #distance (batch_size, mem_slot, que_slot)
    prob_dis = softmax(distance*strengths, 1)
    return torch.transpose(prob_dis, 2, 1).contiguous()

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)



def match_tensor(out, refer_shape):
    
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col        
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col      
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0), mode='reflect')
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]
    
    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row), mode='reflect')
    else:
        crop_row = row - skiprow   
        left_crop_row  = crop_row // 2
        
        right_row = left_crop_row + skiprow
        
        out = out[:,:,left_crop_row:right_row, :]
    return out


def down_up(inputs, depth = 2):
    # inputs should be (N, C, H, W)
    #down path
    inputs = to_variable(inputs, requires_grad=False, var=True,volatile=True)
    org_dim = len(inputs.size())
    if org_dim == 2:
        inputs = inputs.unsqueeze(0).unsqueeze(0)
    
    size_pool = []
    this_out = inputs
    for idx in range(depth):
        size_pool.append(this_out.size()[2::])
        this_out = F.max_pool2d(this_out, kernel_size = 2)
           
    for didx in range(depth):
        this_size = size_pool[depth - didx - 1]
        this_out = torch.nn.UpsamplingBilinear2d(size=this_size)(this_out)
    if org_dim == 2:
        this_out = this_out[0,0]
    return this_out

def Activation(nchan=0, activ=True):
    if activ is 'elu':
        return nn.ELU(inplace=True)
    elif activ is 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activ is 'relu':
        return nn.ReLU(inplace=True)
    elif activ is 'selu':
        print('using selu')
        return nn.SELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class GeneralLinear(nn.Linear):
    def forward(self, inputs):
        inputs_size = inputs.size()
        last_dim = inputs_size[-1]
        out_size = list(inputs.size())
        out_size[-1] = self.out_features

        flat_input = inputs.view(-1, last_dim)
        flat_output = super(GeneralLinear, self).forward(flat_input)

        return flat_output.view(*out_size)

class Dense(nn.Module):
    def __init__(self,  input_dim, out_dim=None, activ='selu'):
        super(Dense, self).__init__()
        if out_dim is None:
            out_dim = input_dim
        self.act = Activation(out_dim, activ = activ)
        self.linear = GeneralLinear(input_dim, out_dim)
        
    def forward(self, x):
        out = self.act(self.linear(x))
        return out

class DenseStack(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, depth=0, activ='selu'):
        super(DenseStack, self).__init__()
        self.__dict__.update(locals())
        layers = [Dense(input_dim, hid_dim, activ=activ)]
        for _ in range(depth):
            layers.append(Dense(hid_dim, activ=activ))

        layers.append(GeneralLinear(hid_dim,out_dim))
        self.transform =  nn.Sequential(*layers)
    
    def forward(self,x):
        return self.transform(x)  

def spp(inputs, size_pool=[4]):
    #inputs should be of B*C*row*col
    # the output should be B*C*feat_dim
    feat_dim = np.sum([n**2 for n in size_pool])
    B, C, _, _ = inputs.size()
    output_list = []
    for ks in size_pool:
        this_out = F.adaptive_avg_pool2d(inputs, ks)
        this_out = this_out.view(B, C, ks**2)
        output_list.append(this_out)
    out_tensor = torch.cat(output_list, dim = 2)
    return out_tensor

class LayerNormal(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-9):
        super(LayerNormal, self).__init__()

        self.eps = eps
        self.mu = nn.Parameter(torch.ones(1, d_hid, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, d_hid, 1, 1), requires_grad=True)

    def forward(self, z):
        _ndim = len(z.size())

        batch_size = z.size()[0]
        z_2d = z.view(batch_size, -1)

        mu   = torch.mean(z_2d, 1, keepdim=True)
        sigma =  torch.std(z_2d, 1, keepdim=True )
        
        #print(mu.size(), sigma.size(), z.size(),flat_z.size())
        
        if _ndim == 4: 
            mu = mu.unsqueeze(-1).unsqueeze(-1)
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)
            #print(mu.size(), sigma.size(), z.size())
            ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
            ln_out = ln_out * self.mu.expand_as(ln_out) + self.beta.expand_as(ln_out)
        else:
            ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
            mu = self.mu.squeeze(-1).squeeze(-1)
            beta = self.beta.squeeze(-1).squeeze(-1)
            ln_out = ln_out * mu.expand_as(ln_out) + beta.expand_as(ln_out)

        return ln_out

class ConvBN(nn.Module):
    def __init__(self, inChans, outChans, activ='lrelu', dilation=1, kernel_size = 3, norm='bn'):
        super(ConvBN, self).__init__()
        
        redu = dilation*(kernel_size - 1)
        p1 = redu//2
        p2 = redu - p1

        self.norm = getNormLayer(norm)(inChans)
        self.act  = Activation(inChans, activ = activ) #if use_activ is True else passthrough()
        self.conv = padConv2d(inChans, outChans, kernel_size=kernel_size, padding= None, dilation=dilation)

    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        return out


class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans, activ= 'lrelu', norm='bn'):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()
        self.norm = getNormLayer(norm)(inputChans)
        self.act  = Activation(inputChans,activ=activ)

        self.conv  = padConv2d(inputChans, outChans, kernel_size=3, padding=1)
        #self.conv = padConv2d(inputChans, outChans, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        #out = self.norm(self.act(self.conv(x)))
        out = self.norm(x)
        out = self.act(out)
        out = self.conv(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, activ='lrelu', norm = 'bn', pooling= False):
        super(DownTransition, self).__init__()
        self.norm  = getNormLayer(norm)(inChans)
        self.act1  = Activation(inChans,activ=activ)

        if pooling:
            self.down_conv = padConv2d(inChans, outChans, kernel_size=3, padding=1, stride=1)
            self.max_pooling = nn.MaxPool2d(kernel_size = 2, stride=2)
        else:
            self.down_conv = padConv2d(inChans, outChans, kernel_size=3, padding=1, stride=2)
            self.max_pooling = passthrough()
        
        #self.drop  = nn.Dropout2d(dropout) if dropout else passthrough()
        #self.conv_ops = _make_nConv(outChans, outChans, nConvs, activ=activ, norm=norm)
        _layers  = [ConvBN(outChans, outChans, activ=activ) for _ in range(nConvs)]
        self.conv_ops = nn.Sequential(*_layers)
        #self.act2 = Activation(outChans,activ=activ)
        
    def forward(self, x):
        xx = self.act1(self.norm(x))
        xx   = self.max_pooling(self.down_conv(xx))
        #down = self.act1(self.norm(xx))
        #out = self.drop(down)
        out = self.conv_ops(xx)
        out = xx + out
        return out

class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, catChans=None, 
                 dropout=False,stride=2,activ='lrelu', norm='bn'):
        # remeber inChans is mapped to hidChans, then concate together with skipx, the composed channel = outChans
        super(UpConcat, self).__init__()
        #hidChans = outChans // 2
        self.outChans = outChans
        #self.drop1   = nn.Dropout2d(dropout) if dropout else passthrough()
        #self.drop2   = nn.Dropout2d(dropout) if dropout else passthrough()
        self.norm = getNormLayer(norm)(inChans)
        self.act1 = Activation(inChans, activ= activ)

        self.up_conv = nn.ConvTranspose2d(inChans, hidChans, kernel_size=3, 
                                          padding=1, stride=stride, output_padding=1)
        
        

        self.reshape = ConvBN(catChans + hidChans, outChans, activ=activ)
        _layers = [ConvBN(outChans, outChans, activ=activ) for _ in range(nConvs-1)]
        self.conv_ops = nn.Sequential(*_layers)

        #self.conv_ops = _make_nConv(catChans + hidChans, self.outChans, depth=nConvs, activ = activ, norm=norm)
        #self.act2 = Activation(outChans, activ= activ)

    def forward(self, x, skipx):
        #out = self.drop1(x)
        #skipxdo = self.drop2(skipx)
        x = self.act1(self.norm(x))
        out  = self.up_conv(x)
        out  = match_tensor(out, skipx.size()[2:])
        xcat = torch.cat([out, skipx], 1)
        
        xcat  =  self.reshape(xcat)
        out   = self.conv_ops(xcat)
        out  += xcat
        return out

        
class UpConv(nn.Module):
    def __init__(self, inChans, outChans, dropout=False, stride = 2,activ= True, norm='bn'):
        super(UpConv, self).__init__()
        #hidChans = outChans // 2
        self.norm = getNormLayer(norm)(inChans)
        self.act1 = Activation(inChans,activ= activ)

        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=3, 
                                          padding=1, stride = stride, output_padding=1)

    def forward(self, x, dest_size):
        '''
        dest_size should be (row, col)
        '''
        #out = self.drop1(x)
        x = self.act1(self.norm(x))
        out = self.up_conv(x)
        out = match_tensor(out, dest_size)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans=1, hidChans=16, activ= 'lrelu', norm='bn'):
        super(OutputTransition, self).__init__()
        
        self.conv1  = ConvBN(inChans,  hidChans, activ=activ)

        self.conv2  = ConvBN(hidChans, outChans, activ=activ)
        
    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv2(self.conv1(x))
        return out

class ConvGRU_cell(nn.Module):
    """Initialize a basic Conv GRU cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c
      filter_size: int thself.up_tr256_12   = UpConv(256, 256, 2)at is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
    """
    def __init__(self, input_chans, filter_size, output_chans, dropout = None, activ='selu'):
        super(ConvGRU_cell, self).__init__()
        
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.output_chans = output_chans
        #self.batch_size=batch_size
        self.dropout = dropout
        self.padding=(filter_size-1)//2 #in this way the output has the same size
        self.register_buffer('device_id', torch.zeros(1))
        self.conv1_inp = padConv2d(input_chans, 3*output_chans,  kernel_size=filter_size,  padding=self.padding, stride=1)
        self.conv1_out = padConv2d(output_chans, 3*output_chans, kernel_size=filter_size, padding=self.padding, stride=1)
        
        self.act1  = Activation(output_chans, activ= activ)

        #self.norm1 = LayerNormal(3*output_chans)
        #self.norm2 = LayerNormal(3*output_chans)
    
    def get_dropmat(self, batch_size):
        W_dropmat = None
        U_dropmat = None
        if self.dropout is not None and self.training is True:
            droprate = 1- self.dropout
            W_dropmat = to_device(torch.bernoulli( droprate * torch.ones(batch_size, self.input_chans, 1, 1)), self.device_id)
            U_dropmat = to_device(torch.bernoulli( droprate * torch.ones(batch_size, self.output_chans, 1, 1)), self.device_id)

        return [W_dropmat, U_dropmat]

    def forward(self, inputs, h_tm1, spatial_gate, dropmat= None):
        # input (B, T, W, H), hidden_state ()
        if  dropmat is not None:
            droprate = 1-self.dropout
            W_drop, U_drop = dropmat
            if self.training is True and W_drop is not None and U_drop is not None:
               inputs  = inputs  *  W_drop.expand_as(inputs) / droprate
               h_tm1   = h_tm1  *  U_drop.expand_as(h_tm1) / droprate

        input_act  = self.conv1_inp(inputs)
        hidden_act = self.conv1_out(h_tm1) 

        (a_inp_r,a_inp_i,a_inp_n) = torch.split(input_act,  self.output_chans,dim=1)#it should return 3 tensors
        (a_hid_r,a_hid_i,a_hid_n) = torch.split(hidden_act, self.output_chans,dim=1)#it should return 3 tensors
        if spatial_gate:
            mean_r_inp = (a_inp_r + a_hid_r)
            mean_r_inp = torch.mean(torch.mean(mean_r_inp, -1, keepdim =True), -2, keepdim =True)

            mean_i_inp = (a_inp_i + a_hid_i)
            mean_i_inp = torch.mean(torch.mean(mean_i_inp, -1, keepdim =True), -2, keepdim =True)

            #print('mean_r_inp shape is: ', mean_r_inp.size())
            r = torch.sigmoid(mean_r_inp).expand_as(a_inp_r)
            i = torch.sigmoid(mean_i_inp).expand_as(a_hid_i)
            n = self.act1(a_inp_n + r * a_hid_n)
            h_t = (1- i)*n  + i*h_tm1 
            
        else:
            r = torch.sigmoid(a_inp_r + a_hid_r)
            i = torch.sigmoid(a_inp_i + a_hid_i)
            n = self.act1(a_inp_n + r * a_hid_n)
            h_t = (1- i)*n  + i*h_tm1 
        return h_t

    def init_hidden(self,batch_size, rowsize, colsize):
        return torch.zeros(batch_size,self.output_chans,rowsize, colsize)
