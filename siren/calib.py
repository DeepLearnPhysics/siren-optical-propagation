import torch
import numpy as np

from siren.models import Siren, weighted_mse_loss
from photonlib import PhotonLib

def collate_fn(batch):
    cat = lambda x : np.squeeze(x) if len(batch) <= 1 else np.concatenate(x)
    output = {}
    keys = batch[0].keys()
    for key in keys:
        output[key] = torch.as_tensor(
            cat([data[key] for data in batch])
        )
    return output

class SirenCalib(Siren):
    def __init__(self, cfg, name='siren', plib='photonlib'):
        super().__init__(cfg, name, plib)
        
        calib_cfg = self.model_cfg['calib']
        self.coord_offset = torch.tensor(
            calib_cfg.get('coord_offset', [0,0,0]),
            
        )

        err0 = calib_cfg.get('err0')
        if err0 is None:
            self.err0 = torch.scalar_tensor(0.)
        elif isinstance(err0, str):
            self.err0 = torch.tensor(np.load(err0))
        else:
            self.err0 = torch.tensor(err0)

        self._init_light_yield(calib_cfg)

    def _init_light_yield(self, calib_cfg):
        requires_grad = not calib_cfg.get('fix_light_yield', True)
        light_yield_input = calib_cfg.get('light_yield')
        
        if light_yield_input is not None:
            if isinstance(light_yield_input, str):
                light_yield = np.load(light_yield_input)
            else:
                light_yield = light_yield_input
        elif 'n_pmts' in calib_cfg:
            light_yield = np.ones(calib_cfg['n_pmts'])
        else:
            light_yield = 1
            
        self.light_yield = torch.nn.Parameter(
            torch.tensor(np.nan_to_num(light_yield), dtype=torch.float32),
            requires_grad=requires_grad,
        )

    def get_siren_output(self, coords, clone=True):
        return super().forward(coords, clone)

    def forward(self, batch, clone_coords=True):
        # apply offet to match lut coordinates
        coords = batch['charge_coord'] 
        coords = coords + self.coord_offset.to(coords.device)
        
        # TODO(2022-11-08 kvt) Replace with clamp function
        coords = self.meta.norm_coord(coords)
        coords[coords<-1] = -1
        coords[coords>1] = 1
        
        vis, coords_out = self.get_siren_output(coords, clone_coords)

        # inv. transform of SIREN output, if needed
        vis = self.to_vis(vis)
                        
        charge_size = batch['charge_size']
        toc = np.concatenate([[0], np.cumsum(charge_size.to('cpu'))])
        
        evt_mask = charge_size > 0
        sel = torch.where(evt_mask)[0]
        
        pred = []
        q = batch['charge_value']
        for i, evt in enumerate(sel):
            start, stop = toc[evt:evt+2]
            sum_q_vis = (q[start:stop,None] * vis[start:stop]).sum(axis=0)
            pred.append(sum_q_vis)
            
        pred_pe = torch.stack(pred) * self.light_yield

        output = {
            'pred' : pred_pe,
            'coords' : coords_out,
            'evt_mask' : evt_mask,
            'vis' : vis,
            'batch_size' : len(pred),
        }
        return output
            

    def chi2_loss(self, obs, pred, reduce=torch.mean):
        err0 = self.err0.to(pred.device)
        if err0.dim() == 1:
            err0 = err0[None,:]

        weight = 1 / (pred + err0**2)

        obs = obs.squeeze()
        pred = pred.squeeze()
        weight = weight.squeeze()

        mask = ~torch.isnan(obs)

        loss = weighted_mse_loss(
            pred[mask], obs[mask], weight[mask],
            reduce=reduce,
        )
        return loss


    def training_step(self, batch, batch_idx):        

        output = self(batch)

        evt_mask = output['evt_mask']
        obs = batch['light_value'][evt_mask]
        pred = output['pred']

        loss = self.chi2_loss(obs, pred)
        self.log('loss', loss, batch_size=output['batch_size'])

        return loss

    def validation_step(self, batch, batch_idx):        

        output = self(batch)

        evt_mask = output['evt_mask']
        obs = batch['light_value'][evt_mask]
        pred = output['pred']

        loss = self.chi2_loss(obs, pred)
        self.log('val_loss', loss, batch_size=output['batch_size'])
        return loss
