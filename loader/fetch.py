import os
import json 
from pprint import pprint
from pathlib import Path

# remove extension
available_datasets = [f.split('.')[0] for f in os.listdir('config/dataset/')]

def fetch_dataset(dataset, config_num):
    if not dataset in available_datasets:
        raise ValueError(f'Dataset {dataset} not found. Only {available_datasets} are available.')
    else:
        config_num = str(config_num).zfill(3)
        dataset_path = f'config/dataset/{dataset}/{config_num}.json'
        with open(dataset_path) as f:
            dataset_config = json.load(f)

        #pprint(f'Fetching dataset {dataset} with config: {dataset_config}')
    if dataset == "taps_norm":
        from loader.taps import TactileDataset
        
        train_dataset = TactileDataset(f"{dataset_config['path']}/train", features='pol')
        val_dataset = TactileDataset(f"{dataset_config['path']}/val", features='pol')
        test_dataset = TactileDataset(f"{dataset_config['path']}/test", features='pol')
        return dataset_config, (train_dataset, val_dataset, test_dataset)
    
    elif dataset == "random_circle":
        from loader.random_circle_loader import RandomCircleDataset

        print(Path(f"{dataset_config['path']}/{config_num}/train").resolve())

        train_dataset = RandomCircleDataset(f"{dataset_config['path']}", config_num, "train")
        val_dataset = RandomCircleDataset(f"{dataset_config['path']}", config_num, "val")
        test_dataset = None

        return dataset_config, (train_dataset, val_dataset, test_dataset)