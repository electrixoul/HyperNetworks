"""
HyperNetworks GPU Implementation - Core Modules

This file contains the core HyperNetwork implementation for GPU acceleration.
The HyperNetwork generates weights for layers in the primary network dynamically.

Original Paper: "HyperNetworks" by Ha, Dai and Schmidhuber (2016)
https://arxiv.org/abs/1609.09106

This implementation is optimized for GPU execution with direct CUDA tensor operations.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class HyperNetwork(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(), 2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(), 2))

    def forward(self, z):
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
