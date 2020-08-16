import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class MLPN(nn.ModuleList):
    def __init__(self, last_norm, dropout=0.1, input_dim=640):
        super(MLPN, self).__init__()
        dim = 512
        norm='none'
        activ = 'lrelu'
        self.last_norm = last_norm
        self.MLP = []
        self.MLP += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        self.MLP += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.MLP += [LinearBlock(dim, input_dim, norm='none', activation='none')]
        self.MLP = nn.Sequential(*self.MLP)

        if last_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim)
        elif last_norm == 'in':
            self.norm = nn.InstanceNorm1d(input_dim)
        elif last_norm == 'ln':
            self.norm = LayerNorm(input_dim)
        elif last_norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [bs, feat]
        """
        residual = x
        output = self.MLP(x)
        output = self.dropout(output)
        output = output + residual
        if self.last_norm == 'none':
            return output
        output = self.norm(output)
        return output
