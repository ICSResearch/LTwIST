import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from itertools import repeat
import torch.nn.functional as F
import collections.abc as container_abcs
from torch.nn.modules.module import Module
from .layers import unetConv2, unetUp
from utils.util import init_weights, count_param
import models


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):  
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


class UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=1, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [32, 64, 128, 256]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.center = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # upsampling
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*512
        maxpool1 = self.maxpool(conv1)  # 16*256*256

        conv2 = self.conv2(maxpool1)  # 32*256*256
        maxpool2 = self.maxpool(conv2)  # 32*128*128

        conv3 = self.conv3(maxpool2)  # 64*128*128
        maxpool3 = self.maxpool(conv3)  # 64*64*64

        center = self.center(maxpool3)  # 256*32*32
        up3 = self.up_concat3(center, conv3)  # 64*128*128
        up2 = self.up_concat2(up3, conv2)  # 32*256*256
        up1 = self.up_concat1(up2, conv1)  # 16*512*512

        final = self.final(up1)

        return final


class Conv(Module):
    def __init__(self, config, ic, oc):
        super(Conv, self).__init__()
        self.config = config
        self.ic = ic
        self.oc = oc
        self.w = nn.Parameter(torch.Tensor(oc, ic, 9))
        self.padding = _pair(1)
        self.init = nn.Parameter(torch.zeros([ic, 9, 9], dtype=torch.float32))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))  

    def forward(self, inputs):
        init = self.init + torch.eye(9, dtype=torch.float32).unsqueeze(0).repeat((self.ic, 1, 1)).to(self.config.device)
        weight = torch.reshape(torch.einsum('abc, dac->dab', init, self.w), (self.oc, self.ic, 3, 3))
        outputs = F.conv2d(inputs, weight, None, 1, self.padding)
        return outputs


class pre_layer(nn.Module):
    def __init__(self, config):
        super(pre_layer, self).__init__()

        self.unet = UNet()


    def forward(self, x_recon):

        x_recon = torch.transpose(x_recon, 0, 1).reshape([-1, 1, 32, 32])  
        x_output = self.unet(x_recon)

        x_output = torch.transpose(x_output.reshape(-1, 1024), 0, 1)   
        return x_output


class post_layer(nn.Module):
    def __init__(self, config):
        super(post_layer, self).__init__()

        self.unet = UNet()

        self.conv_out = nn.Sequential(
            nn.ReLU(),
            Conv(config, 32, 32),
            Conv(config, 32, 32),
            Conv(config, 32, 1),
        )

    def forward(self, x_recon):
        x_output = self.unet(x_recon)
        return x_output


class HybridNet(nn.Module):
    def __init__(self, config):
        super(HybridNet, self).__init__()
        self.config = config
        self.phi_size = 32
        points = self.phi_size ** 2
        phi_init = np.random.normal(0.0, (1 / points) ** 0.5, size=(int(config.ratio * points), points))      
        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=False)
        self.Q = nn.Parameter(torch.from_numpy(np.transpose(phi_init)).float(), requires_grad=False)

        self.num_layers = 6
        self.pre_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.pre_block.append(pre_layer(config))

        self.post_block = nn.ModuleList()
        for i in range(self.num_layers):
            self.post_block.append(post_layer(config))

        self.threshold = nn.Parameter(torch.Tensor([0.01]), requires_grad=False)
        self.weights = []
        self.etas = []
        self.alphas = []
        self.betas = []

        alp = 0.9
        bet = 0.8

        for i in range(self.num_layers):
            self.weights.append(nn.Parameter(torch.tensor(1.), requires_grad=False))
            self.register_parameter("eta_" + str(i + 1), nn.Parameter(torch.tensor(0.1), requires_grad=False))  
            self.etas.append(eval("self.eta_" + str(i + 1)))
            self.register_parameter("alpha_" + str(i + 1), nn.Parameter(torch.tensor(alp), requires_grad=False)) 
            self.alphas.append(eval("self.alpha_" + str(i + 1)))
            self.register_parameter("beta_" + str(i + 1), nn.Parameter(torch.tensor(bet), requires_grad=False))
            self.betas.append(eval("self.beta_" + str(i + 1)))

    def forward(self, inputs):
        batch_size = inputs.size(0)     
        # inputs = torch.unsqueeze(inputs,0)  
        y = self.sampling(inputs, self.phi_size)
        recon = self.recon(y, self.phi_size, batch_size)    
        return recon

    def sampling(self, inputs, init_block):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=2), dim=0)
        inputs = torch.reshape(inputs, [-1, init_block ** 2])
        inputs = torch.transpose(inputs, 0, 1)
        y = torch.matmul(self.phi, inputs)
        return y

    def recon(self, y, init_block, batch_size):
        idx = int(self.config.block_size / init_block)

        recon = torch.matmul(self.Q, y)   
        xp = recon
        x = recon
        for i in range(self.num_layers):
            if i != 0:
                x = (1-self.alphas[i]) * xp + (self.alphas[i]-self.betas[i]) * recon
                xp = recon
            recon = recon - self.weights[i] * torch.mm(torch.transpose(self.phi, 0, 1), (torch.mm(self.phi, recon) - y))
            recon = recon - self.pre_block[i](recon)  
            recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])  
            recon = torch.mul(torch.sign(recon), F.relu(torch.abs(recon) - self.threshold))
            recon = recon - self.post_block[i](recon)
            recon = torch.reshape(recon, [-1, init_block ** 2])
            recon = torch.transpose(recon, 0, 1) 

            if i != 0:
                recon = x + self.betas[i] * recon


        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])    # [9,1,32,32]
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)    #[3,1,96,32]
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)  #[1,1,96,96]
        return recon

