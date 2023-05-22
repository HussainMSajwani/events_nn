import torch
from torch_geometric.data import Data, Batch

class BPTTGraphDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        return [self.data[idx+i] for i in range(self.seq_len)]
    

class BPTTGraphLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size, seq_len, shuffle=False, **kwargs):
        bptt_dataset = BPTTGraphDataset(dataset, seq_len)
        super().__init__(bptt_dataset, batch_size, shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, data):
        batch_size = len(data)
        graphs_per_pred = len(data[0])
        make_batch = lambda list_of_graphs: Batch.from_data_list(list_of_graphs)
        return [
            make_batch([data[i][j] for i in range(batch_size)]) 
            for j in range(graphs_per_pred)
            ]