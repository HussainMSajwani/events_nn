import os
import json 

from models.TactiGraph import TactiGraph
from models.eViT import eViT

available_models = {
    "tactigraph": TactiGraph,
    "evit": eViT
}

def fetch_model(model_name, model_config):
    model_path = os.path.join('config', 'model', model_name, f'{model_config}'.zfill(3) + '.json')
    with open(model_path) as f:
        model_args = json.load(f)
    try:
        model = available_models[model_name](**model_args)
        return model
    except KeyError:
        raise ValueError(f'Model {model} not found. Only {available_models.keys()} are available.')