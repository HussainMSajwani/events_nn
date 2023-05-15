import os
import json 
from pprint import pprint

# remove extension
available_datasets = [f.split('.')[0] for f in os.listdir('config/dataset/')]

def fetch_dataset(dataset):
    if not dataset in available_datasets:
        raise ValueError(f'Dataset {dataset} not found. Only {available_datasets} are available.')
    else:
        with open(f'config/dataset/{dataset}.json') as f:
            dataset_config = json.load(f)

        #pprint(f'Fetching dataset {dataset} with config: {dataset_config}')
    if dataset == "taps_norm":
        from loader.taps import TactileDataset

        train_dataset = TactileDataset(f"{dataset_config['path']}/train", features='pol')
        val_dataset = TactileDataset(f"{dataset_config['path']}/val", features='pol')
        test_dataset = TactileDataset(f"{dataset_config['path']}/test", features='pol')
        return dataset_config, (train_dataset, val_dataset, test_dataset)