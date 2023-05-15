import pytorch_lightning as pl
from torch_geometric.data import Data
import torch

class contact_angle_module(pl.LightningModule):

    def __init__(self, model, batch_size=1):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def forward(self, data : Data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        data = batch
        output = self.model(data)
        loss = self.loss(output, data.y)
        self.log_dict(
            {'train_loss': loss, 
             'train_loss_degrees': torch.rad2deg(loss) 
             }, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data = batch
        output = self.model(data)
        loss = self.loss(output, data.y)
        self.log_dict({
            'val_loss': loss,
            'val_loss_degrees': torch.rad2deg(loss)
            }, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss(self, output, target):
        print(output, target)
        return torch.nn.functional.l1_loss(output, target)
    
