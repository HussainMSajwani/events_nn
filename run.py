from loader.random_circle import RandomCircleDataset
dataset = RandomCircleDataset('./data/random_circle/processed1')
dataset.process()
print(len(dataset))
print(dataset[0])