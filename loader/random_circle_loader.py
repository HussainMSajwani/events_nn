from pathlib import Path
from typing import Union

import torch
import json
import h5py
import dask.array as da
import pandas as pd
import numpy as np
from .loader_utils import events_in_circle, downsample_events, iter_periods
from utils.config import get_sensor_config

from tqdm.auto import tqdm
from scipy.sparse import coo_matrix
from utils.make_graph import make_graph

#TODO: cehck if processed
#TODO: save params

class RandomCircleDataset(torch.utils.data.Dataset):

    def __init__(
            self, 
            outdir: Union[str, Path], 
            config: str = '001', 
            subset: str = 'train',
            dt: float = 100, # ms
            downsampling_factor: int = 1
            ):
        super().__init__()

        self.outdir = Path(outdir).resolve() / config / subset 
        self.outdir.mkdir(parents=True, exist_ok=True)

        with open(f'config/dataset/random_circle/{config}.json') as f:
            self.config = json.load(f)
        
        self.subset = subset
        self.bag_dir = Path(self.config['path'])/ 'bag'

        self.dt = dt
        self.downsampling_factor = downsampling_factor
        

    def ev_arr_iter(self):
        events = da.array(h5py.File(self.bag_dir / f'{self.subset}_events.h5', 'r')['events'])
        df = pd.read_csv(self.bag_dir / f'{self.subset}_df.csv')

        sensor_config = get_sensor_config(self.config['sensor_name'])
        events, roi_size = events_in_circle(events, sensor_config['circle_radius'], sensor_config['center'])

        # Downsample events
        events = downsample_events(events, self.downsampling_factor)

        # iterate over periods
        period_length = 5 # s
        assert (period_length / self.dt*1e3).is_integer(), f'Period length must be a multiple of dt. {period_length} is not a multiple of {self.dt}'
        full_length = (events[-1, 2] - events[0, 2]).compute()
        iterator = tqdm(iter_periods(events, period_length), total = full_length // (period_length*1e9), desc='Periods')
        for i, period in enumerate(iterator):
            t0, t1 = period[0, 2].compute(), period[-1, 2].compute()
            #print(period.shape, t0, t1, (t1 - t0)*1e-9)
            # Split period into smaller segments of length dt
            iterator2 = iter_periods(period, self.dt/1000)#, total = period_length*1e3 // self.dt, desc='Segments')
            for j, segment in enumerate(iterator2):
                t0, t1 = segment[0, 2].compute(), segment[-1, 2].compute()
                t = (t1 + t0) / 2
                df_idx = np.searchsorted(df['ts'], t)
                if df_idx >= len(df):
                    #print(f'Index {df_idx} is out of bounds. Skipping...')
                    continue
                y = df.iloc[df_idx][['contact_angle_x', 'contact_angle_y']]
                yield segment, y

    def process(self):
        for j, (ev_arr, y) in enumerate(self.ev_arr_iter()):
            #print(ev_arr.shape, y)
            if not (self.outdir / f'sample_{j:05}.pt').exists():
                data = make_graph(ev_arr, y)
                torch.save(data, self.outdir / f'sample_{j:05}.pt')


    def to_img(self, ev_arr):
        sensor_config = get_sensor_config(self.config['sensor_name'])
        row = ev_arr[:, 0].compute()
        col = ev_arr[:, 1].compute()
        pol = 255*np.ones_like(ev_arr[:, 3].compute())#ev_arr[:, 3].compute()

        H, W = sensor_config['frame_size']
        img = coo_matrix((pol, (row, col)), shape=(H, W)).toarray().astype('uint8')
        return img
    
    def __getitem__(self, idx):
        return torch.load(self.outdir / f'sample_{idx:05}.pt')
    
    def __len__(self):
        return len(list(self.outdir.glob('*.pt')))
