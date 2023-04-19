import torch
from torch_geometric.nn import SplineConv

class eViT(torch.nn.Module):

    def __init__(self, image_size, patch_size) -> None:
        super().__init__()
        self.embd_conv1 = SplineConv(1, 8, dim=3, kernel_size=5)
        self.embd_conv2 = SplineConv(8, 8, dim=3, kernel_size=5)
        self.embd_conv3 = SplineConv(8, 16, dim=3, kernel_size=5)
        self.embd_conv4 = SplineConv(16, 16, dim=3, kernel_size=5)

        self.patch_size = patch_size
        self.image_size = image_size
        
    def to_patches(self, data):
        pass

    def forward(self, data):
        pass


