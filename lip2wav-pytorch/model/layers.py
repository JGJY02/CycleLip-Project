import torch
import torch.nn as nn
from torch.nn import functional as F
from hparams import hparams as hps

import sys

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)
    
class HighwayNet:
    def __init__(self, in_dim, out_dim, name=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scope = "HighwayNet" if name is None else name
        
        self.H_layer = nn.sequential(
            LinearNorm(self.in_dim, self.out_dim),
            nn.ReLU()
        )
        self.T_layer = nn.sequentail(
            LinearNorm(self.in_dim, self.out_dim),
            nn.Sigmoid()
        )
        
    
    def __call__(self, inputs):
        H = self.H_layer(inputs)
        T = self.T_layer(inputs)
        return H * T + inputs * (1. - T)
    

    
class custom_conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(custom_conv1d, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        
        self.batched = torch.nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.dropout = torch.nn.Dropout(hps.tacotron_dropout_rate)

    def forward(self, signal):
        conv_signal = self.dropout(self.activation(self.batched(self.conv(signal))))
        return conv_signal

class Conv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding_type=None, bias=True, residual=False, prev_dim = None, w_init_gain = 'relu'):
        super(Conv3d, self).__init__()
        padding = 0
        
        if padding_type == 'multi_stride':
            
            output_height = (prev_dim + stride[1] - 1) // stride[1]
            output_width = (prev_dim + stride[2] - 1) // stride[2]

            dim1 = max(0, ((hps.T - 1) * stride[0] - hps.T + kernel_size) // 2)
            dim2 = max(0, ((output_height - 1) * stride[1] - prev_dim + kernel_size) // 2)
            dim3 = max(0, ((output_width - 1) * stride[2] - prev_dim + kernel_size) // 2)
            padding = tuple([dim1,dim2,dim3])

        else:
            dim1 = ((hps.T - 1) * stride - hps.T + kernel_size) // 2
            dim2 = ((prev_dim - 1) * stride - prev_dim + kernel_size) // 2
            dim3 = ((prev_dim - 1) * stride - prev_dim + kernel_size) // 2
            padding = tuple([dim1,dim2,dim3])


        # print("Current Padding used is {}".format(padding))
        self.conv = torch.nn.Conv3d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding,
                                    bias=bias)
        

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        
        self.conv_block = nn.Sequential(
            self.conv,
            nn.BatchNorm3d(out_channels)
        )
        self.residual = residual
        self.act = nn.ReLU()

    def forward(self, x):
        # print("Currently allocated {} gb of data from layers".format(torch.cuda.memory_allocated()/ (1024**3)))
        
        out = self.conv_block(x)
        # print(out.shape)
        if self.residual:
            out += x
            
        return self.act(out)
    



class ConvolutionEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvolutionEncoder, self).__init__()
    
        self.Convolution_Encoder_Blocks = nn.ModuleList([
        nn.Sequential(
            Conv3d(3, 32, kernel_size = 5, stride = (1, 2, 2), padding_type = 'multi_stride', prev_dim = 128),
            Conv3d(32, 32, kernel_size = 3, stride = 1, residual = True, prev_dim = 64),
            Conv3d(32, 32, kernel_size = 3, stride = 1, residual = True, prev_dim = 64)
        ),
        nn.Sequential(
            Conv3d(32, 64, kernel_size = 3, stride = (1, 2, 2), padding_type = 'multi_stride', prev_dim = 64),
            Conv3d(64, 64, kernel_size = 3, stride = 1, residual = True, prev_dim = 32),
            Conv3d(64, 64, kernel_size = 3, stride = 1, residual = True, prev_dim = 32)
        ),
        nn.Sequential(
            Conv3d( 64, 128, kernel_size = 3, stride = (1, 2, 2), padding_type = 'multi_stride', prev_dim = 32),
            Conv3d(128, 128, kernel_size = 3, stride = 1, residual = True, prev_dim = 16),
            Conv3d(128, 128, kernel_size = 3, stride = 1, residual = True, prev_dim = 16)
        ),
        nn.Sequential(
            Conv3d(128, 256, kernel_size = 3, stride = (1, 2, 2), padding_type = 'multi_stride', prev_dim = 16),
            Conv3d(256, 256, kernel_size = 3, stride = 1, residual = True, prev_dim = 8),
            Conv3d(256, 256, kernel_size = 3, stride = 1, residual = True, prev_dim = 8),
        ),
        nn.Sequential(
            Conv3d(256, 512, kernel_size = 3, stride = (1, 2, 2), padding_type = 'multi_stride', prev_dim = 8),
            Conv3d(512, 512, kernel_size = 3, stride = 1, residual = True, prev_dim = 4),
            Conv3d(512, 512, kernel_size = 3, stride = 1, residual = True, prev_dim = 4),
            Conv3d(512, 512, kernel_size = 3, stride = (1, 4, 4), padding_type = 'multi_stride', prev_dim = 4)
        )

        ])
            
            
    def forward(self, signal):
        conv_signal = self.Convolution_Encoder_Blocks(signal)
        return conv_signal