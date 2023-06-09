from models.TactiGraph_lstm import TactiGraph_lstm
from loader.bptt import BPTTGraphLoader
from loader.random_circle_loader import RandomCircleDataset
from utils.lightning_modules import contact_angle_module_bptt
import pytorch_lightning as pl

batch_size = 5
seq_len = 10


train_dataset = RandomCircleDataset('data/random_circle/', subset='train', config='001')
val_dataset = RandomCircleDataset('data/random_circle/', subset='val', config='001')

train_loader = BPTTGraphLoader(train_dataset, batch_size=batch_size, shuffle=True, seq_len=seq_len)
val_loader = BPTTGraphLoader(val_dataset, batch_size=1, shuffle=False, seq_len=seq_len)

model = TactiGraph_lstm()
trainer_module = contact_angle_module_bptt(model, batch_size=batch_size, seq_len=seq_len, lr=1e-4)

trainer = pl.Trainer(gpus=1, max_epochs=100, logger=pl.loggers.CSVLogger('logs', name='bptt'))
trainer.fit(trainer_module, train_loader, val_loader)
