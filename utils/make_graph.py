import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, knn_graph
import torch_geometric.transforms as T

#TODO: automate dimensionality
im_height=260
im_width=346


def make_graph(ev_arr, y):
    events = ev_arr.compute()

    coord1, coord2 = torch.tensor(events[:, 0:2].astype(np.float32)).T 
    ts = events[:, 2]
    ts = ((ts - ts.min()) / (ts.max() - ts.min())).astype(np.float32)
    coord3 = torch.tensor(ts)
    pos = torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T

    feature = torch.tensor(2*events[:, 3].astype(np.float32)-1)
    feature = feature.view(-1, 1)

    edge_index = radius_graph(pos, r=0.05, max_num_neighbors=32, )

    y = torch.tensor(np.array(y, dtype=np.float32)).reshape(1, -1)

    data = Data(x=feature, edge_index=edge_index, pos=pos, y=y)
    
    transforms = T.Compose([T.Cartesian(cat=False, norm=True)
    #, T.LargestConnectedComponents(1)
    ])
    data = transforms(data)
    return data