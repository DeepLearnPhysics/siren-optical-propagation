import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader
from photonlib import PhotonLib, Meta

class PhotonLibWrapper(Dataset):
    def __init__(self, plib_cfg):
        self.plib = PhotonLib.load(**plib_cfg)

    def __len__(self):
        return len(self.plib)
    
    def __getitem__(self, idx):
        output = dict(
            voxel_id=idx,
            vis=self.plib[idx],
        )

        return output

class PhotonLibPartition(Dataset):

    def __init__(self, partition, **kwargs):
        self.plib = PhotonLib.load(**kwargs)
        self.partition = Meta(partition, self.plib.meta.ranges)

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, i_entry):
        idx = self.partition.voxel_to_idx(i_entry)

        size, mod = np.divmod(self.plib.meta.shape,
                              self.partition.shape)

        offset = lambda i : (i < mod) * i + (i >= mod) * mod

        padding_0 = -1 * (idx > 0) 
        padding_1 = idx < self.partition.shape - 1
                
        start = idx * size + offset(idx) + padding_0
        stop = (idx+1) * size + offset(idx+1) + padding_1
        ranges = np.column_stack([start, stop])

        mgrid = np.meshgrid(*[np.arange(*r) for r in ranges], indexing='ij')
        idx_sel = np.column_stack([g.flatten() for g in mgrid])
        vox_id = self.plib.meta.idx_to_voxel(idx_sel)

        shape = np.empty(4, dtype=int)
        shape[:3] = np.diff(ranges, axis=1).astype(int).flat
        shape[-1] = self.plib.n_pmts

        padding = -np.column_stack([padding_0, padding_1])

        return {
            'idx':      i_entry,
            'voxel_id': vox_id,
            'vis':      self.plib.vis[vox_id],
            'ranges':   ranges,
            'padding':  padding,
            'shape':    shape,
        }

    @classmethod
    def create(cls, cfg, name='dataset'):
        ds_cfg = cfg[name]
        partition = ds_cfg['partition']
        plib_kwargs = ds_cfg['photonlib']
        return cls(partition, **plib_kwargs)


class PhotonLibPatch(Dataset):
    def __init__(self, plib_cfg):
        self.plib = PhotonLib.load(**plib_cfg)

    def __len__(self):
        return len(self.plib)

    def __getitem__(self, i_entry):
        meta = self.plib.meta

        idx = meta.voxel_to_idx(i_entry)

        idx[idx==0] += 1
        idx[idx==meta.shape-1] -= 1

        mgrid = np.meshgrid(*[range(i-1, i+2) for i in idx], indexing='ij')
        idx_cube = np.column_stack([g.flatten() for g in mgrid])
        vox_ids = meta.idx_to_voxel(idx_cube)

        return {
            'voxel_ids': vox_ids,
            'vis':       self.plib.vis[vox_ids],
            'mask':      vox_ids == i_entry,
        }

class PhotonLibPmtPair(Dataset):
    def __init__(self, plib_cfg):
        self.plib = PhotonLib.load(**plib_cfg)

    @property
    def mid(self):
        return self.plib.n_pmts // 2

    def __len__(self):
        return len(self.plib) 

    def __getitem__(self, i_entry):
        vis = self.plib.vis[i_entry].reshape(2, -1).T

        coord = np.empty((self.mid, 5), dtype=np.float32)
        coord[:,:3] = self.plib.meta.voxel_to_coord(i_entry, norm=True)
        coord[:,3:] = self.plib.pmt_pos_norm[:self.mid,1:]

        output = {
            'idx':      i_entry,
            'vis':      vis,
            'coord':    coord,
        }
        return output

class SirenCalibDataset(Dataset):
    def __init__(self, filepath, adc2pe=None, tpc=None, light_idx=None):
        self._files = filepath 
        self._tpc = tpc
        self._light_idx = light_idx
        self._adc2pe = adc2pe

        evt_cnts = []
        for fpath in self._files:
            with h5py.File(fpath, 'r') as f:
                evt_cnts.append(len(f['charge/size']))
        
        self._file_toc = np.insert(np.cumsum(evt_cnts), 0, [0]).astype(int)
        self._evt_toc = {}
        
    def __len__(self):
        return self._file_toc[-1]
    
    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError('index', i, 'out of range')
            
        file_idx = np.digitize(i, self._file_toc) - 1
    
        fpath = self._files[file_idx]
        evt_idx = i - self._file_toc[file_idx]
        
        with h5py.File(fpath, 'r') as f:
            toc = self._evt_toc.get(file_idx)
            if toc is None:
                toc = np.insert(f['charge/size'][:].cumsum(), 0, [0])
                self._evt_toc[file_idx] = toc
            
            hits = f['charge/data'][toc[evt_idx]:toc[evt_idx+1]]
            light_value = f['light/data'][evt_idx]
        
        if self._adc2pe is not None:
            light_value *= self._adc2pe

        if self._tpc is not None:
            mask = hits['mask']
            tpc = hits['tpc']
            mask &= tpc == self._tpc
            hits = hits[mask]

        if self._light_idx is not None:
            s = slice(*self._light_idx)
            light_value = light_value[s]
            
        coords = np.column_stack([hits['x'], hits['y'], hits['z']])
        output = dict(
            charge_coord=coords.astype(np.float32),
            charge_value=hits['q'].astype(np.float32),
            charge_tpc=hits['tpc'],
            charge_mask=hits['mask'],
            light_value=light_value.astype(np.float32),
        )
        return output

def dataloader_factory(cls, cfg):
    dataset = cls(**cfg['dataset'])
    dataloader = DataLoader(dataset, **cfg['dataloader'])
    return dataloader
