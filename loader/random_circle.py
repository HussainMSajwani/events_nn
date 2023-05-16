import torch
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional
import h5py
import dask.array as da
import pandas as pd
import json
from utils.config import get_sensor_config
from tqdm.auto import tqdm

from .base import BaseDataset
from .loader_utils import events_in_circle, downsample_events, iter_periods, iter_periods

class RandomCircleDataset(BaseDataset):
    """Random circle dataset.
    """

    def __init__(self, outdir: Union[str, Path], split: str = 'train'):
        self.outdir = Path(outdir).resolve()
        self.split = split
        rawdir = Path('./data/random_circle/') / self.split / 'raw' 
        super(RandomCircleDataset, self).__init__(raw_dir=rawdir)

        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
            params = {
                'split': self.split,
                'processed': False
            }
            with open(self.outdir / 'params.json', 'w') as f:
                json.dump(params, f, indent=4)
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
            downsampling_factor: int = 1,
            sensor_name: str = 'multifunctional_v2'
            ):
        """Process the dataset.
        """
        assert not self.params['processed'], 'Dataset already processed.'
        # Get sensor config
        sensor_config = get_sensor_config(sensor_name)
        
        # Load events
        h5 = h5py.File(self.raw_dir / 'events.h5', 'r')
        events = da.from_array(h5['events'])

        # Filter events in circle
        events, roi_size = events_in_circle(events, sensor_config['circle_radius'], sensor_config['center'])

        # Downsample events
        events = downsample_events(events, downsampling_factor)

        # W, H = roi_size
        # assert W // patch_size and H // patch_size, 'Patch size must be a divisor of ROI size.'
        
        # pos = events[:, 0:2]
        # patch_idx = (pos[:, 0] // patch_size) * W // patch_size + (pos[:, 0] // patch_size)

        # # iterate over periods
        # period_length = 5 # s
        # full_length = (events[-1, 2] - events[0, 2]).compute()
        # iterator = tqdm(iter_periods(events, period_length), total = full_length // (period_length*1e9), desc='Periods')
        # for i, period in enumerate(iterator):
        #     t0, t1 = period[0, 2].compute(), period[-1, 2].compute()
        #     #print(period.shape, t0, t1, (t1 - t0)*1e-9)
        #     # Split period into smaller segments of length dt
        #     iterator2 = tqdm(iter_periods(period, dt/1000), total = period_length*1e3 // dt, desc='Segments')
        #     for j, segment in enumerate(iterator2):
        #         t0, t1 = segment[0, 2].compute(), segment[-1, 2].compute()
        #         #print(segment.shape, t0, t1, (t1 - t0)*1e-9)
        #     #print()
        
if __name__ == '__main__':
    dataset = RandomCircleDataset('../data/random_circle/processed')
    dataset.process()
