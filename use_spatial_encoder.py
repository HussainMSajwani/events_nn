#%%
from loader.taps import TactileDataset

dataset = TactileDataset('/home/hussain/me/projects/events_nn/data/morethan3500ev_lessthan_9deg/train', features='pol')

#%%
d = dataset[0].cuda()
d
# # %%
# from utils.extract_patches import grid_patch
# mask = grid_patch(d, 10, 170, 170)
# mask
# %%
from models.eViT import eViT

model = eViT(170, 10).cuda()
model
# %%
out = model(d)
print(out.shape)
# %%
