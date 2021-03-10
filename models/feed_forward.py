import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from configuration import *

class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # 2 blocks
        output = self.model(x)
        output = self.model(output)
        return (x + output) # residual connection

class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.input_shape = 17*2
        self.output_shape = 17*3
        self.block = LinearModule()
        self.linear_in = nn.Linear(self.input_shape, 1024)
        self.liner_out = nn.Linear(1024, self.output_shape)
        self.batchnorm = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        o = self.linear_in(x)
        o = self.batchnorm(o)
        o = self.relu(o)
        o = self.block(o)
        o = self.block(o)
        o = self.liner_out(o)
        return o
