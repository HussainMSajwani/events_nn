import torch
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, raw_dir: Union[str, Path]):
        super(BaseDataset, self).__init__()
        self.raw_dir = Path(raw_dir).resolve()
        assert self.raw_dir.exists(), f'Raw data directory {self.raw_dir} does not exist.'

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    

    