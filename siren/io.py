import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from photonlib import PhotonLib

class PhotonLibDataset(Dataset):
    def __init__(self, filepath):
        self.plib = PhotonLib.load_file(filepath)

    def __len__(self):
        return len(self.plib)
    
    def __getitem__(self, idx):
        output = dict(
            voxel_id=idx,
            vis=self.plib[idx],
        )
        return output

class DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return idx

def dataloader_factory(cfg):
    dataset = PhotonLibDataset(**cfg['photonlib'])
    dataloader = DataLoader(dataset, **cfg['dataloader'])
    #dataset = DummyDataset()
    #dataloader = DataLoader(dataset)
    return dataloader
