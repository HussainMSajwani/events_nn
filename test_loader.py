# %%
from loader.random_circle_loader import RandomCircleDataset
from loader.bptt import BPTTGraphLoader

dataset = RandomCircleDataset('data/random_circle/', subset='val', config='001')
# %%
loader = BPTTGraphLoader(dataset, batch_size=2, shuffle=False, seq_len=5)

data = next(iter(loader))
data
# %%
from models.TactiGraph_lstm import TactiGraph_lstm

model = TactiGraph_lstm()
output = model(data)
# %%
output.shape

# %%
