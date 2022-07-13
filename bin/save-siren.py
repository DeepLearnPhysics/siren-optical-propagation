#!/usr/bin/env python3 

import yaml
import torch
import h5py
import fire

from functools import partial
from tqdm import trange

from siren import Siren
from photonlib import PhotonLib


def save(cfg_file, ckpt_file, outfile='siren_pred.h5', device=0):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    model = Siren.load_from_checkpoint(ckpt_file, cfg=cfg)
    model.eval()
    model.to(device)

    meta = model.meta
    nx, ny, nz = meta.shape
    n_pmts = cfg['siren']['network']['out_features']

    to_vis = partial(model.inv_transform, lib=torch)

    vis_pred = torch.empty(
        nx, ny, nz, n_pmts, dtype=torch.float32, device=device
    )

    for ix in trange(nx):
        torch.cuda.empty_cache()
        x = meta.idx_to_coord(meta.idx_at('x', ix, device=device), norm=True) 
        pred, coord = model(x)

        vis_pred[ix] = to_vis(pred).reshape(ny, nz, n_pmts).detach()

    PhotonLib.save(outfile, vis_pred.to('cpu'), meta)

if __name__ == '__main__':
    fire.Fire(save)
