import torch
import h5py
import numpy as np
from pytorch_lightning.core.lightning import LightningModule

from siren.networks import Siren as SirenNet
from photonlib import PhotonLib, Meta

def gradient(y, x, grad_outputs=None, **kwargs):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)

    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True, **kwargs
    )

    return grad[0]

class Sobel3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        g = torch.tensor([1,2,1]).float()  # gaussian filter
        d = torch.tensor([-1,0,1]).float() # finite diff.
        
        # set weight for gradient filters
        gx = d[:,None,None] * g[None,:,None] * g[None,None,:]
        gy = g[:,None,None] * d[None,:,None] * g[None,None,:]
        gz = d[None,None,:] * g[None,:,None] * g[:,None,None]

        pars = torch.cat([gx.unsqueeze(0), gy.unsqueeze(0), gz.unsqueeze(0)])
        pars = pars.unsqueeze(1)
        
        self.sobel = torch.nn.Conv3d(
            in_channels=1, out_channels=3, kernel_size=3, stride=1, bias=False, 
             padding_mode='replicate', padding=1,
        )
        
        self.sobel.weight = torch.nn.Parameter(pars, requires_grad=False)
    
    def forward(self, img):
        return self.sobel(img)

def get_sobel_coeffs():
    g = torch.tensor([1,2,1]).float()
    d = torch.tensor([-1,0,1]).float()

    coeffs = torch.stack(
        (d[:,None,None] * g[None,:,None] * g[None,None,:],
         g[:,None,None] * d[None,:,None] * g[None,None,:],
         g[:,None,None] * g[None,:,None] * d[None,None,:])
    ).reshape(3,-1)

    return coeffs

def weighted_mse_loss(input, target, weight=1.):
    loss = (weight * (input - target)**2).mean()
    return loss

class Siren(LightningModule):
    def __init__(self, cfg, name='siren', plib='photonlib'):
        super().__init__()

        self.model_cfg = cfg[name]
        self._lr0 = self.model_cfg.get('lr', 5e-5)
        self._bias_threshold = self.model_cfg.get('bias_threshold', 4.5e-5)
        self._weighting_scheme = self.model_cfg.get('weight', None)
        self._hardsigmoid = self.model_cfg.get('hardsigmoid', False)

        net_pars = self.model_cfg['network']
        self.net = SirenNet(**net_pars)

        self.plib_cfg = cfg[plib]
        plib_fpath = self.plib_cfg['filepath']
        self.meta = Meta.load(plib_fpath, lib=torch)

        if self.plib_cfg.get('transform', False):
            self.tranform, self.inv_transform = PhotonLib.partial_transform(
                vmax=self.plib_cfg.get('vmax', 1),
                eps=self.plib_cfg.get('eps', 1e-7),
                sin_out=self.plib_cfg.get('sin_out', False)
            )

        #self.sigmoid = torch.nn.Hardsigmoid()

        # validate config
        sin_out = self.plib_cfg.get('sin_out', False)
        linear_out = net_pars['outermost_linear']
        assert sin_out is not linear_out, 'mismatch sin_out, outermost_linear'

    def forward(self, x):
        out, coords = self.net(x)

        if self._hardsigmoid:
            out = torch.nn.functional.hardsigmoid(out)

        return out, coords

    def training_step(self, batch, batch_idx):
        voxel_id = batch['voxel_id']
        target = batch['vis']

        x = self.meta.voxel_to_coord(voxel_id, norm=True)
        pred, coord = self(x)

        # do inverse transform (if needed) for bias metric
        if self.plib_cfg.get('transform', False):
            target_orig = self.inv_transform(target, lib=torch)
            pred_orig = self.inv_transform(pred.detach(), lib=torch)
        else:
            target_orig = target
            pred_orig = pred.detach()

        # calcuate L2 loss
        # weight = target_orig * 1e6
        # weight[weight==0] = 1
        if self._weighting_scheme == 'vis':
            weight = target_orig / self.plib_cfg.get('eps', 1e-7)
            weight[weight==0] = 1
        else:
            weight = 1.

        loss = weighted_mse_loss(pred, target, weight)
        self.log('loss', loss, on_step=False, on_epoch=True)

        mask = target_orig > self._bias_threshold
        if torch.any(mask):
            a = pred_orig[mask]
            b = target_orig[mask]
            bias = 2 * torch.abs((a - b) / (a + b)).mean() * 100
            self.log('bias', bias, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self._lr0)
        return opt

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.1, patience=50, threshold=1e-4, 
            min_lr=1e-8, threshold_mode='rel', cooldown=10, verbose=True)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler' : sch,
                'interval'  : 'epoch',
                'monitor'   : 'bias',
            },
        }
