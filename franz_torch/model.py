from typing import Tuple, Collection

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import embedding, LongTensor


class CloudModel(nn.Module):
    def __init__(
            self,
            field_dims,
            output_dim,
            embed_dim,
            y_range,
            dropout,
            dnn_dims,
            input_depth,
            cnn_dims,
            conv_kernel_size,
            pool_kernel_size
    ):
        super(CloudModel, self).__init__()
        self.y_range = y_range

        size = embed_dim
        for i in range(0, len(cnn_dims)):
            size = round((size - conv_kernel_size[0]) / pool_kernel_size[0])
        self.cnn_output_size = size * size * cnn_dims[-1]

        n_users = field_dims[0]
        n_items = field_dims[1]
        n_factors = embed_dim
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users, 1), (n_items, 1)
        ]]

        self.embed_output_dim = len(field_dims) * embed_dim
        self.cnn = CNN(input_depth, cnn_dims, dropout, conv_kernel_size, pool_kernel_size)
        self.dnn = DNN(n_factors*9 + self.cnn_output_size + 5, output_dim, dnn_dims, dropout)
        self.stdlin = StandardLinear(input_size=1, output_size=1, dropout=0.5)
        self.enhance = None

    def forward(self, users: LongTensor, items: LongTensor):
        # Get embeddings
        u, v = self.u_weight(users), self.i_weight(items)

        # Extract correlations
        ## Euclidean Distance

        euclid_squared, euclid = minkovski(u, v, 2)
        minkovski_squared_3, minkovski_3 = minkovski(u, v, 3)
        minkovski_squared_4, minkovski_4 = minkovski(u, v, 4)
        minkovski_squared_5, minkovski_5 = minkovski(u, v, 5)

        u_squared = u.square()
        v_squared = v.square()

        sin_u = u.sin()
        sin_v = v.sin()
        cos_u = u.cos()
        cos_v = v.cos()

        pythagoras = (u_squared + v_squared).sqrt()

        # Inner product
        inner = u * v
        inner = inner.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        inner = inner.unsqueeze(-1)
        #inner = self.stdlin(inner)

        # Outer product
        outer = torch.einsum('bp, bq->bpq', u, v).unsqueeze(1)

        # CNN
        cnn = self.cnn(outer)

        # Concatenate outputs
        concat = torch.cat(
            (
                u,
                v,
                inner,
                euclid_squared,
                euclid,
                u_squared,
                v_squared,
                sin_u,
                sin_v,
                cos_u,
                cos_v,
                pythagoras,
                self.u_bias(users),
                self.i_bias(items),
                cnn
            ),
            1)

        # DNN
        dnn = self.dnn(concat)

        # Last step
        x = dnn #+ inner
        if self.y_range is not None:
            x = torch.sigmoid(x) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return torch.flatten(x)


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, dnn_dims, dropout, output_layer=True):
        super(DNN, self).__init__()
        layers = list()
        for dnn_dim in dnn_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dnn_dim, bias=True))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.PReLU())
            layers.append(nn.BatchNorm1d(num_features=dnn_dim))
            input_dim = dnn_dim
        if output_layer:
            layers.append(nn.Linear(in_features=input_dim, out_features=output_dim))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)


class CNN(nn.Module):
    def __init__(self, input_depth, cnn_dims, dropout, conv_kernel_size, pool_kernel_size):
        super(CNN, self).__init__()
        layers = list()
        for output_depth in cnn_dims:
            layers.append(nn.Conv2d(
                in_channels=input_depth,
                out_channels=output_depth,
                kernel_size=conv_kernel_size,
                bias=True))
            layers.append(nn.MaxPool2d(
                kernel_size=pool_kernel_size,
            ))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.PReLU())
            layers.append(nn.BatchNorm2d(num_features=output_depth))
            input_depth = output_depth
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class StandardLinear(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(StandardLinear, self).__init__()
        self.perceptron = nn.Linear(in_features=input_size, out_features=output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(num_features=output_size)

    def forward(self, x):
        x = self.perceptron(x)
        x = self.prelu(x)
        x = self.bn(x)
        return x


def minkovski(u: torch.Tensor, v: torch.Tensor, p: int = 1):
    tmp = (u - v).pow(p).sum(1)
    return tmp.unsqueeze(-1), tmp.pow(1/p).unsqueeze(-1)
