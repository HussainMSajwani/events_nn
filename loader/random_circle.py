import torch
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional
import h5py
import dask.array as da
import pandas as pd
import json

from utils.config import get_sensor_config

from .base import BaseDataset
from .utils import events_in_circle, downsample_events

class RandomCircleDataset(BaseDataset):
    """Random circle dataset.
    """

    def __init__(self, outdir: Union[str, Path], split: str = 'train'):
        super(RandomCircleDataset, self).__init__(Path('../data/random_circle/raw').resolve())

        self.outdir = Path(outdir).resolve()
        self.split = split

        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            params = {
                'split': self.split,
                'processed': False
            }
            with open(self.outdir / 'params.json', 'w') as f:
                json.dump(params, f)
    @property
    def params(self):
        with open(self.outdir / 'params.json', 'r') as f:
            params = json.load(f)
        return params
    
    def __len__(self):
        assert self.params['processed'], 'Dataset not processed.'
        return len(list(self.outdir.glob('*.h5')))
    
    def __getitem__(self, index):
        assert self.params['processed'], 'Dataset not processed.'
        with h5py.File(self.outdir / f'{index}.h5', 'r') as f:
            return f['image'][:], f['pose'][:]
        

    def process(
            self,
            dt: float = 30, # ms
            patch_size: Tuple[int, int] = (5, 5),
            downsampling_factor: int = 1,
            sensor_name: str = 'multifunctional_v2'
            ):
        """Process the dataset.
        """
        assert not self.params['processed'], 'Dataset already processed.'
        # Get sensor config
        sensor_config = get_sensor_config(sensor_name)
        
        # Load events
        h5 = h5py.File(self.rawdir / self.split / 'events.h5', 'r')
        events = da.from_array(h5['events'])

        # Filter events in circle
        events, roi_size = events_in_circle(events, sensor_config['circle_radius'], sensor_config['center'])

        # Downsample events
        events = downsample_events(events, downsampling_factor)

        W, H = roi_size
        assert W // patch_size[0] and H // patch_size[1], 'Patch size must be a divisor of ROI size.'
        
        pos = events[:, 0:2]
        patch_idx = (pos[:, 0] // P) * W // P + (pos[:, 0] // P)

        
        

