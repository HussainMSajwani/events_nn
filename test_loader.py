# %%
from loader.random_circle_loader import RandomCircleDataset

if __name__ == '__main__':
    dataset = RandomCircleDataset('data/random_circle/', subset='val', config='001')
    
    print(dataset[0])

# %%
