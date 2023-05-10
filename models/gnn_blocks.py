import torch.nn as nn
import torch_geometric.nn as gnn

class SplineBlock(nn.Module):

    def __init__(self, d_in, d_out, activation=nn.ELU(), dim=3, kernel_size=2) -> None:
        super().__init__()

        self.conv = gnn.SplineConv(d_in, d_out, dim, kernel_size)
        self.activation = activation
        self.bn = gnn.BatchNorm(d_out)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.activation(x)
        x = self.bn(x)
        return x

