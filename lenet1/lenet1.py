"""
Author: Raphael Senn

Syntax:
    units       : neurons
    links       : connections between units (neurons)
    free_params : learnable parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

class LeNet1(nn.Module):
    """
    Reproduction of the Convolutional Neural Network of the Paper:
    Backpropagation Applied to Handwritten Zip Code Recognition from 1989. 
    """ 
    def __init__(self) -> None:
        super(LeNet1, self).__init__()

        units = []          # keep track of units
        links = 0           # keep track of links
        free_params = 0     # keep track of free parameters (learnable parameters)

        # initialize weights
        winit = lambda F_in, shape: Uniform(-2.4/F_in, 2.4/F_in).sample(shape)

        """
        input shape [1, 1, 16, 16]                  -> 16 * 16 = 256 units
        12 kernels with shape [5, 5]                -> 12 * 5 * 5 = 300 free parameters
        12 feature maps                             -> 12 * 8 * 8 = 768 units
        total of (12 * 8 * 8) * (5 * 5) + 768 = 19968 connections (768 biases)
        total of 300 + 768 = 1068 learnable parameters (300 weight params, 768 biases)
        """ 
        self.H1w = nn.Parameter(winit(1 * 5 * 5, (12, 1, 5, 5)))    # 300 learnable params
        self.H1b = nn.Parameter(torch.zeros((12, 8, 8)))            # 768 learnable params
        units.extend([256, 768])
        free_params += (12 * 5 * 5) + (12 * 8 * 8)
        links += (12 * 8 * 8) * (5 * 5) + 768

        """
        1 unit in H2.X has 8 * 5 * 5 = 200 inputs from eight of the 8 x 8 feature maps via the 5 x 5 kernel
        therefore 1 unit in H2.X has 200 weights and 1 bias
        total of (12 * 4 * 4) * (8 * 5 * 5) + 192 = 38592 connections (192 units * 200 weight params + 192 biases)
        total of 192 biases + 12 * 200 weight params = 2592 learnable parameters
        """ 
        self.H2w = nn.Parameter(winit(8 * 5 * 5, (12, 8, 5, 5)))    # 2400 learnable params
        self.H2b = nn.Parameter(torch.zeros((12, 4, 4)))            # 192 learnable params
        units.append(192)
        free_params += 200 * 12 + 192
        links += (8 * 5 * 5) * (12 * 4 * 4) + 192

        """
        gets cat(H2.1.flatten(), ..., H2.12.flatten()) as input
        resulting in 12 * 4 * 4 = 192 hidden units
        they are fully connected to another 30 units
        total of 192 * 30 + 30 = 5780 connections
        total of 192 * 30 + 30 = 5780 learnable parameters
        """ 
        self.H3w = nn.Parameter(winit(192, (30, 192)))              # 5760 learnable params
        self.H3b = nn.Parameter(torch.zeros((30,)))                 # 30 learnable params
        units.append(30)
        free_params += 192 * 30 + 30
        links += 192 * 30 + 30

        """
        fully connected
        total of 30 * 10 + 10 = 310 connections
        total of 30 * 10 + 10 = 310 learnable parameters
        """ 
        self.fcw = nn.Parameter(winit(30, (10, 30)))                # 300 learnable params
        self.fcb = nn.Parameter(torch.zeros((10,)))                 # 10 learnable params
        units.append(10) 
        free_params += 30 * 10 + 10
        links += 30 * 10 + 10

        self.units = units
        self.free_params = free_params                              # 9760 learnable params
        self.links = links

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # input                                                         # [1, 1, 16, 16]
        # NOTE: manually adding padding=2, for background = -1.0 
        x = F.pad(x, pad=(2, 2, 2, 2), value=-1.0)
        out = F.conv2d(x, self.H1w, stride=2) + self.H1b                # [1, 12, 8, 8]
        out = F.tanh(out)

        # this was not well explained in the paper, inspired by @karpathy
        out = F.pad(out, pad=(2, 2, 2, 2), value=-1.0)
        out1 = F.conv2d(
            out[:, 0:8], self.H2w[0:4], stride=2) + self.H2b[0:4]       # [1, 4, 4, 4]

        out2 = F.conv2d(
            out[:, 4:12], self.H2w[4:8], stride=2) + self.H2b[4:8]      # [1, 4, 4, 4]

        out3 = F.conv2d(
            torch.cat((out[:, 0:4], out[:, 8:12]), dim=1),
            self.H2w[8:12], stride=2) + self.H2b[8:12]                  # [1, 4, 4, 4]

        out = torch.cat((out1, out2, out3), dim=1)                      # [1, 12, 4, 4]
        out = F.tanh(out)

        out = out.flatten(start_dim=1)                                  # [1, 192]

        out = F.linear(out, self.H3w, self.H3b)                         # [1, 30]
        out = F.tanh(out)

        out = F.linear(out, self.fcw, self.fcb)                         # [1, 10]
        out = F.sigmoid(out)

        return out

    def __str__(self) -> str:
        stats = 'Stats from LeNet-1\n'
        stats += f'total units:              {sum(self.units)}\n'
        stats += f'total connections:        {self.links}\n'
        stats += f'independent parameters:   {self.free_params}\n'
        return stats