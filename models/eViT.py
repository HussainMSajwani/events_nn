import torch
from .gnn_blocks import SplineBlock
from .attn_blocks import Attention
from utils.extract_patches import grid_patch
from torch_scatter import scatter

#TODO: generalize this to work with any number of layers
class eViT(torch.nn.Module):

    def __init__(self, image_size, patch_size, patch_encoder_block = 'spline', d_enc = 8, n_heads = 8, dim_head = 64) -> None:
        super().__init__()

        self.FeatureEncoder = torch.nn.Linear(1, d_enc)

        self.patch_size = patch_size
        self.image_size = image_size
        
        if patch_encoder_block == 'spline':
            block = SplineBlock
        else:
            raise NotImplementedError
        
        self.PatchEncoder = torch.nn.ModuleList(
            [
                block(d_enc, d_enc*2),
                block(d_enc*2, d_enc*2),
                block(d_enc*2, d_enc*4)
            ]
        )

        self.mha = Attention(d_enc*4, heads = n_heads, dim_head = dim_head)

    def forward(self, data):
        data = grid_patch(data, self.patch_size, self.image_size, self.image_size)
        x = self.FeatureEncoder(data.x)
        #x = self.PatchEncoder(x, data.edge_index, data.edge_attr)
        for layer in self.PatchEncoder:
            x = layer(x, data.edge_index, data.edge_attr)

        x = scatter(x, data.node_patch_map, dim=0, reduce='max')
        x = x[None, ...] # add empty batch dimension. TODO: generalize this to work with batch_size > 1
        x = self.mha(x)
        x = x.mean(dim=1)
        return x




