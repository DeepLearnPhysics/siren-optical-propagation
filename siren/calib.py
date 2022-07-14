import torch
import numpy as np

from siren.models import Siren, weighted_mse_loss

def collate_fn(batch):
    output = {}
    keys = batch[0].keys()
    for key in keys:
        output[key] = torch.as_tensor(
            np.concatenate([data[key] for data in batch])
        )
    return output

class SirenCalib(Siren):
    def __init__(self, cfg, name='siren', plib='photonlib'):
        super().__init__(cfg, name, plib)
        
        calib_cfg = self.model_cfg['calib']
        self.coord_offset = torch.tensor(
            calib_cfg.get('coord_offset', [0,0,0]),
            
        )
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
        
    def forward(self, batch):
        coords = batch['charge_coord'].clone()
        
        # apply offet to match lut coordinates
        coords += self.coord_offset.to(coords.device)
        
        coords = self.meta.norm_coord(coords)
        coords[coords<-1] = -1
        coords[coords>1] = 1
        
        vis, coords_out = super().forward(coords)

        if self.plib_cfg.get('transform', False):
            vis = self.inv_transform(vis, lib=torch)
                
        #self.log('vmin', vis.min(), prog_bar=True)  
        #self.log('vmax', vis.max(), prog_bar=True)
        
        charge_size = batch['charge_size']
        toc = np.concatenate([[0], np.cumsum(charge_size.to('cpu'))])
        
        evt_mask = charge_size > 10
        sel = torch.where(evt_mask)[0]
        
        pred = []
        q = batch['charge_value']
        for i, evt in enumerate(sel):
            start, stop = toc[evt:evt+2]
            sum_q_vis = (q[start:stop,None] * vis[start:stop]).sum(axis=0)
            pred.append(sum_q_vis)
            
        pred_pe = torch.stack(pred) * self.light_yield
        return pred_pe, coords_out, evt_mask
            
    def training_step(self, batch, batch_idx):        

        pred, __, evt_mask = self(batch)
        
        target = batch['light_value'][evt_mask]
        light_mask = ~torch.isnan(target)
        light_mask &= target > 1

        loss = weighted_mse_loss(
            pred[light_mask], target[light_mask], 1/pred[light_mask]
        )
        #self.log('loss', loss, on_step=False, on_epoch=True)
        self.log('loss', loss)
        
        return loss
