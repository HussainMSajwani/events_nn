from torch_geometric.data import Data
from torch_geometric.transforms import LargestConnectedComponents
import torch

def grid_patch(data : Data, patch_size : int, max_x : int, max_y : int, extract_lcc : bool = True):
    """
    Returns a list of a node and edges mapping such that each node and edge are assigned to a patch in the frame
    """
    floor_pos = torch.div(data.pos, patch_size, rounding_mode='floor')
    node_patch_map = floor_pos[:, 0] * (max_x // patch_size) + floor_pos[:, 1] 
    edges_in_same_patch = node_patch_map[data.edge_index[0]] == node_patch_map[data.edge_index[1]]
    valid_edges = data.edge_index[:, edges_in_same_patch]
    edge_patch_map = node_patch_map[valid_edges[0]]

    if extract_lcc:
        lcc = LargestConnectedComponents()
        data = lcc(data)

    floor_pos = torch.div(data.pos, patch_size, rounding_mode='floor')
    node_patch_map = floor_pos[:, 0] * (max_x // patch_size) + floor_pos[:, 1] 
    edges_in_same_patch = node_patch_map[data.edge_index[0]] == node_patch_map[data.edge_index[1]]
    valid_edges = data.edge_index[:, edges_in_same_patch]
    valid_edge_attr = data.edge_attr[edges_in_same_patch]
    edge_patch_map = node_patch_map[valid_edges[0]]
    
    out = Data(x=data.x, edge_index=valid_edges, pos=data.pos, y=data.y, edge_attr=valid_edge_attr)
    out.node_patch_map = node_patch_map.long()
    out.edge_patch_map = edge_patch_map.long()

    return out.to(data.x.device)



