#%%
from loader.taps import TactileDataset

dataset = TactileDataset('/home/hussain/me/projects/events_nn/data/morethan3500ev_lessthan_9deg/train', features='pol')

#%%
d = dataset[0]
d
# %%
from utils.extract_patches import grid_patch
grid_patch(d, 10, 170, 170)
# %%
