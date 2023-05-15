import torch

from tqdm.auto import trange, tqdm
import numpy as np

from typing import Callable, List, Optional, Tuple, Union
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, voxel_grid, max_pool

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"

class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int], transform: Callable[[Data, ], Data] = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = list(size)
        self.transform = transform

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        assert edge_index is not None, "edge_index must not be None"

        cluster = voxel_grid(pos[:, :2], batch=batch, size=self.voxel_size)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"


class model2(torch.nn.Module):

    def __init__(self, pooling_size=(16/346, 12/260), pooling_outputs=32, pooling_after_conv2=False, more_layer=False, more_block=False):
        super(model2, self).__init__()
        dim = 3

        bias = False
        root_weight = False

        kernel_size = 2

        self.pooling_size = np.array(pooling_size)
        self.pooling_output = pooling_outputs
        self.pooling_after_conv2 = pooling_after_conv2
        self.more_layer = more_layer
        self.more_block = more_block

        self.conv1 = SplineConv(1, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=8)

        if self.more_layer:
            self.conv1_1 = SplineConv(8, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1_1 = BatchNorm(in_channels=8)

        self.conv2 = SplineConv(8, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=16)

        self.conv2_1 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2_1 = BatchNorm(in_channels=16)
        self.pool2_1 = MaxPooling(self.pooling_size/2, transform=Cartesian(norm=True, cat=False))

        self.conv3 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=16)

        self.conv4 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=16)

        self.conv5 = SplineConv(16, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=pooling_outputs)
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=pooling_outputs)
        self.conv7 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=pooling_outputs)

        if self.more_block:
            self.conv6_1 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm6_1 = BatchNorm(in_channels=pooling_outputs)
            self.conv7_1 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm7_1 = BatchNorm(in_channels=pooling_outputs)


        self.pool_final = MaxPoolingX(0.25, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=2, bias=bias)

    def forward(self, data):
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        if self.more_layer:
            data.x = elu(self.conv1_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1_1(data.x)
        
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        
        data.x = elu(self.conv2_1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2_1(data.x)

        if self.pooling_after_conv2:
            data = self.pool2_1(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)
        
        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        if self.more_block:
            data.x = elu(self.conv6_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm6_1(data.x)
            data.x = elu(self.conv7_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm7_1(data.x)
            data.x = data.x + x_sc


        x = self.pool_final(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

class TactiGraph(model2):

    def __init__(self):
        params = {
            'more_block': True,
            'more_layer': False,
            'pooling_after_conv2': True,
            'pooling_outputs': 32,
            'pooling_size': [0.023121387283236993, 0.011538461538461539]
            }
        super().__init__(**params)

    def forward(self, data):
        return super().forward(data)


