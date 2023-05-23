import argparse
import sys
from pathlib import Path

from loader.fetch import fetch_dataset
from models.fetch import fetch_model

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import torch
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='taps_norm')
parser.add_argument('--dataset_config', type=int, default=1)

parser.add_argument('--model', type=str, default='tactigraph')
parser.add_argument('--model_config', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpus', type=int, default=1)

args = parser.parse_args()


dataset_config, (train_dataset, val_dataset, test_dataset) = fetch_dataset(args.dataset, args.dataset_config)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=dataset_config['seq_len'] == 'short', num_workers=4)
val_loader = DataLoader(val_dataset, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4)

model = fetch_model(args.model, args.model_config)

if dataset_config['task'] == 'contact_angle':
    from utils.lightning_modules import contact_angle_module
    pl_module = contact_angle_module(model, batch_size=args.batch_size, lr=args.lr)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=args.gpus,
    max_epochs=args.epochs,
    logger=pl.loggers.CSVLogger('logs/', name=f'{args.dataset}_{args.model}'),
)

trainer.fit(pl_module, train_loader, val_loader)




