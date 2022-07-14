import h5py
import sys
import os
import importlib
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
    def __init__(
        self, filepath, 
        adc2pe=None, tpc=None, light_idx=None,
        apply_charge_mask=False, chunk_size=1,
    ):
        self._files = filepath 
        self._tpc = tpc
        self._light_idx = light_idx
        self._apply_charge_mask = apply_charge_mask
        self._chunk_size = chunk_size

        self._load_adc2pe(adc2pe)
        self._evt_toc = {}
        self._partition_toc = {}
        self._build_file_toc()

    def _load_adc2pe(self, adc2pe):
        if isinstance(adc2pe, str):
            self._adc2pe = np.load(adc2pe)
        elif adc2pe is None:
            self._adc2pe = None
        else:
            self._adc2pe = np.array(adc2pe, dtype=np.float32)

    @staticmethod
    def build_toc(cnts):
        return np.insert(np.cumsum(cnts), 0, [0]).astype(int)

    def _build_file_toc(self):
        n_evts = []
        for fpath in self._files:
            with h5py.File(fpath, 'r') as f:
                n_evts.append(len(f['charge/size']))

        if self._chunk_size == 1:
            self._file_toc = self.build_toc(n_evts)

        else:
            n_partitions = np.asarray(n_evts) // self._chunk_size
            n_partitions[n_partitions==0] = 1
            self._file_toc = self.build_toc(n_partitions)

    def _decode_idx(self, i):
        file_idx = np.digitize(i, self._file_toc) - 1
        idx = i - self._file_toc[file_idx]

        if file_idx not in self._evt_toc:
            fpath = self._files[file_idx]
            with h5py.File(fpath, 'r' ) as f:
                self._evt_toc[file_idx] = self.build_toc( f['charge/size'][:])

        evt_toc = self._evt_toc[file_idx]

        if self._chunk_size == 1:
            return file_idx, idx, evt_toc[idx:idx+2]

        if file_idx not in self._partition_toc:
            with h5py.File(fpath, 'r' ) as f:
                file_size = len(f['charge/size'])
            n_parts = max(1, file_size//self._chunk_size)
            partition = np.full(n_parts, file_size//n_parts)
            partition[:file_size%n_parts] += 1
            self._partition_toc[file_idx] = self.build_toc(partition)

        part_toc = self._partition_toc[file_idx]
        return file_idx, idx, evt_toc[part_toc[idx:idx+2]]

        
    def __len__(self):
        return self._file_toc[-1]
    
    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError('index', i, 'out of range')
            
        file_idx, idx, hit_ranges = self._decode_idx(i)
        fpath = self._files[file_idx]
        
        with h5py.File(fpath, 'r') as f:
            hits = f['charge/data'][slice(*hit_ranges)]

            if self._chunk_size == 1:
                start, stop = idx, idx+1
            else:
                start, stop = self._partition_toc[file_idx][idx:idx+2]

            light_value = f['light/data'][start:stop]
        
        charge_size = np.diff(self._evt_toc[file_idx][start:stop+1])
        evt_ids = np.repeat(np.arange(len(charge_size)), charge_size)

        if self._tpc is not None:
            tpc = hits['tpc']
            mask = tpc == self._tpc
            hits = hits[mask]
            evt_ids = evt_ids[mask]

        if self._apply_charge_mask:
            mask = hits['mask']
            hits = hits[mask]
            evt_ids = evt_ids[mask]

        charge_size = np.zeros(stop-start, dtype=int)
        ids, cnts = np.unique(evt_ids, return_counts=True)
        charge_size[ids] = cnts

        if self._adc2pe is not None:
            light_value *= self._adc2pe

        if self._light_idx is not None:
            s = slice(*self._light_idx)
            light_value = light_value[:,s]
            
        coords = np.column_stack([hits['x'], hits['y'], hits['z']])
        output = dict(
            file_ids=np.full_like(evt_ids, file_idx),
            evt_ids=evt_ids+start,
            charge_coord=coords.astype(np.float32),
            charge_value=hits['q'].astype(np.float32),
            charge_size=charge_size.squeeze(),
            light_value=light_value.squeeze().astype(np.float32),
        )

        if not self._apply_charge_mask:
            output['charge_mask'] = hits['mask']

        if self._tpc is None:
            output['tpc'] = hits['tpc']

        return output

def dataloader_factory(cls, cfg):
    dataset = cls(**cfg['dataset'])
    dl_cfg = cfg['dataloader'].copy()

    collate_fn_cfg = dl_cfg.get('collate_fn')
    if collate_fn_cfg is not None:
        if collate_fn_cfg.count('.') == 0:
            module = sys.modules['__main__']
            fn_name = collate_fn_cfg
        else:
            module_name, fn_name = os.path.splitext(collate_fn_cfg)
            module = importlib.import_module(module_name)
            fn_name = fname.lstrip('.')

        collate_fn = getattr(module, fn_name)
        dl_cfg['collate_fn'] = collate_fn

    dataloader = DataLoader(dataset, **dl_cfg)
    return dataloader
