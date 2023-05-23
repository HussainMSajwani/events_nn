import pytorch_lightning as pl
from torch_geometric.data import Data
import torch

class contact_angle_module(pl.LightningModule):

    def __init__(self, model, batch_size=1, lr=1e-3):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.lr = lr

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def loss(self, output, target):
        return torch.nn.functional.l1_loss(output, target)
    

class contact_angle_module_bptt(pl.LightningModule):

    def __init__(self, model, batch_size=1, seq_len=1, lr=1e-3):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr

    def forward(self, data : list):
        return self.model(data)
    
    def sequential_loss(self, output, target):
        return torch.nn.functional.l1_loss(output, target)
    
    def training_step(self, batch, batch_idx):
        data = batch
        output = self.model(data)
        y = [data[i].y for i in range(self.seq_len)]
        y = torch.stack(y, dim=0)
        loss = self.sequential_loss(output, y)
        self.log_dict(
            {'train_loss': loss, 
             'train_loss_degrees': torch.rad2deg(loss) 
             }, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data = batch
        output = self.model(data)
        y = [data[i].y for i in range(self.seq_len)]
        y = torch.stack(y, dim=0)
        loss = self.sequential_loss(output, y)
        self.log_dict({
            'val_loss': loss,
            'val_loss_degrees': torch.rad2deg(loss)
            }, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
    

