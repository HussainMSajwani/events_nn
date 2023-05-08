from torch_geometric.data import Data
import torch

def grid_patch(data : Data, patch_size : int, max_x : int, max_y : int):
    """
    Returns a list of a node and edges mapping such that each node and edge are assigned to a patch in the frame
    """
    floor_pos = torch.div(data.pos, patch_size, rounding_mode='floor')
    node_patch_map = floor_pos[:, 0] * (170 // patch_size) + floor_pos[:, 1] 
    edge_patch_map = node_patch_map[data.edge_index[0]] == node_patch_map[data.edge_index[1]]
    return node_patch_map, edge_patch_map



