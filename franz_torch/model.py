import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CloudModel(torch.nn.Module):
    def __init__(
            self,
            field_dims,
            output_dim=1,
            embed_dim=128,
            dropout=0.5,
            dnn_dims=(128, 256, 64, 32, 8),
            input_depth=1,
            cnn_dims=(4, 8, 16, 32)
    ):
        super().__init__()

        self.embedding = FeatureEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim

        self.outerproduct = None
        self.innerproduct = None

        self.cnn = CNN(input_depth, cnn_dims, dropout)
        self.dnn = DNN(self.embed_output_dim, output_dim, dnn_dims, dropout)
        self.enhance = None

    def forward(self, x):
        # Get feature embeddings
        embed_x = self.embedding(x)

        # Joint learning of the wide component and the deep component
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, dnn_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for dnn_dim in dnn_dims:
            layers.append(nn.Linear(in_features=input_dim, out_features=dnn_dim, bias=True))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.BatchNorm1d(num_features=dnn_dim))
            input_dim = dnn_dim
        if output_layer:
            layers.append(nn.Linear(in_features=input_dim, out_features=output_dim))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dnn(x)


class CNN(nn.Module):
    def __init__(self, input_depth, cnn_dims, dropout):
        super().__init__()
        layers = list()
        for output_depth in cnn_dims:
            layers.append(nn.Conv2d(
                in_channels=input_depth,
                out_channels=output_depth,
                kernel_size=(3, 3),
                bias=True))
            layers.append(nn.MaxPool2d(
                kernel_size=(2, 2),
            ))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.BatchNorm1d(num_features=output_depth))
            input_depth = output_depth
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)
