import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.linalg
import math
import numpy as np

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj, device):
        x = torch.permute(x, (0, 2, 3, 1))
        gso = adj.astype(dtype=np.float32)
        gso = torch.tensor(gso)
        gso = gso.to(device)
        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * gso, x_list[k - 1]) - x_list[k - 2])
        x = torch.stack(x_list, dim=2)
        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv
        return cheb_graph_conv

class GCN(nn.Module):
    def __init__(self, n_his):
        super(GCN,self).__init__()
        bias = True
        self.Ks = 3
        self.g1 = ChebGraphConv(1, 32, self.Ks, bias)
        self.g2 = ChebGraphConv(32, 64, self.Ks, bias)
        self.g3 = ChebGraphConv(64, 32, self.Ks, bias)
        self.fc = nn.Linear(n_his * 32, 1)
        self.relu = nn.ReLU()

    def forward(self, x, adj, device):
        x1 = self.relu(self.g1(x, adj, device))
        x1 = torch.permute(x1, (0, 3, 1, 2))
        x2 = self.relu(self.g2(x1, adj, device))
        x2 = torch.permute(x2, (0, 3, 1, 2))
        x3 = self.relu(self.g3(x2, adj, device))
        x3 = torch.permute(x3, (0, 2, 1, 3))
        out = self.fc(x3.reshape((x3.shape[0], x3.shape[1], -1)))

        return out.view(len(out), -1)

class STGCN(nn.Module):
    def __init__(self, n_vertex):
        super(STGCN, self).__init__()
        self.n_vertex = n_vertex
        self.weight = nn.Parameter(torch.FloatTensor(1, n_vertex))
        self.bias = nn.Parameter(torch.FloatTensor(1, n_vertex))
        self.relu = torch.nn.ReLU()

        nn.init.uniform_(self.weight)
        nn.init.uniform_(self.bias)

    def forward(self, x, y):
        device = x.device
        batch_size = x.size(0)
        weight_expanded = self.weight.expand(batch_size, -1).to(device)
        bais_expanded = self.bias.expand(batch_size, -1).to(device)
        res = weight_expanded * x + bais_expanded + y
        res = self.relu(res)
        return res