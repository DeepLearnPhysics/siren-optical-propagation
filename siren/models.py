import torch
import numpy as np

from lightning import LightningModule

from siren.nets import SirenVis
from photonlib import Meta

def weighted_mse_loss(input, target, weight=1., reduce=torch.mean):
    loss = weight * (input - target)**2
    return reduce(loss)

class SirenVisModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.siren = SirenVis(cfg)

        plib_cfg = cfg['photonlib']
        self.meta = Meta.load(plib_cfg['filepath'], lib=torch)

        model_cfg = cfg['model']
        self._lr0 = model_cfg.get('lr', 5e-5)

        self.weight_cfg = model_cfg.get('weight')
        if self.weight_cfg.get('method') == 'vis':
            self.register_buffer(
                'weight_factor', 
                torch.tensor(self.weight_cfg.get('factor', 1.))
            )

        self.bias_threshold = model_cfg.get('bias_threshold', 4.5e-5)
    
    def get_weight_by_vis(self, vis):
        weights = vis * self.weight_factor
        return vis
    
    def get_bias(self, tgt, pred):
        mask = tgt > self.bias_threshold
        a = pred[mask]
        b = tgt[mask]
        bias = (2 * torch.abs(a-b) / (a+b)).mean()
        return bias
                 
    def forward(self, x):
        return self.siren(x)
    
    def training_step(self, batch, batch_idx):
        voxel_id = batch['voxel_id']
        tgt = batch['vis']
        
        # note: input coordinates are normalized to [-1,1]
        x = self.meta.voxel_to_coord(voxel_id, norm=True)
        pred, __ = self.siren(x)
        
        # transform from log-scale to linear-scale, if applicable
        # otherwise inv_transform is just an indentity function
        tgt_linear = self.siren.inv_transform(tgt, lib=torch)
        pred_linear = self.siren.inv_transform(pred.detach(), lib=torch)
        
        # event weighting
        if self.weight_cfg.get('method') == 'vis':
            weights = self.get_weight_by_vis(tgt)
        else:
            weights = 1.
            
        # calculate loss
        loss = weighted_mse_loss(pred, tgt, weights)
        self.log('loss', loss, on_step=False, on_epoch=True)
        
        # calculate bias
        bias = self.get_bias(tgt_linear, pred_linear) * 100. #in percent
        self.log('bias', bias, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self._lr0)
        return opt
