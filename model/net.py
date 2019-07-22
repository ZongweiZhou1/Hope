import torch
from torch_geometric.nn import EdgeConv
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, F_in, F_out):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * F_in, F_out),
            nn.ReLU(inplace=True),
            nn.Linear(F_out, F_out)
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, F_in, F_out):
        super(Block, self).__init__()
        self.edge_conv = EdgeConv(MLP(F_in, F_out), aggr='mean')
        self.mlp = nn.Sequential(nn.Linear(F_out, int(F_out/2)),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(int(F_out/2), F_out))

    def forward(self, x, edge_index):
        x = self.edge_conv(x, edge_index)
        x = F.relu(x)
        x = self.mlp(x)
        return x


class ResGraphNet(nn.Module):
    def __init__(self, F_in=4, F_mid=(64, 64, 64, 128), F_out=256):
        super(ResGraphNet, self).__init__()
        self.block_list = nn.ModuleList()
        F_cat = 0
        for k in F_mid:
            self.block_list.append(Block(F_in, k))
            F_in = k
            F_cat += k

        self.out_layer = nn.Sequential(
            nn.Linear(F_cat, F_out),
            nn.ReLU(inplace=True),
            nn.Linear(F_out, F_out)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        tmp_x = []
        for block in self.block_list:
            x = block(x, edge_index)
            tmp_x.append(x)
            x = F.relu(x)

        x = self.out_layer(torch.cat(tmp_x, dim=1))
        return x


if __name__ == '__main__':
    from torch_geometric.data import Data

    x = torch.tensor([[0.3, 0.4, 0.8, 0.1],
                      [0.4, 0.5, 0.6, 0.2],
                      [0.4, 0.8, 0.4, 0.5],
                      [0.3, 0.2, 0.1, 0.1]])
    edge_index = torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    block = ResGraphNet()
    y = block(data)
    loss = y.sum()
    loss.backward()
    print(y)


