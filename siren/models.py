import torch
import h5py
import numpy as np
import torchmetrics
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

class MaxMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", 
                       default=torch.tensor(0.),
                       dist_reduce_fx=torch.max)

    def update(self, value : float):
        value = torch.as_tensor(value)
        self.value.fill_(
            max(self.value.item(), value.item())
        )

    def compute(self):
        return self.value.item()

class WeightLUT:
    def __init__(self, lut, min, max):
        self._min = min
        self._max = max
        self._lut = lut
    
    def __getitem__(self, x):
        with torch.no_grad():
            x = torch.as_tensor(x)
            lut = torch.as_tensor(self._lut).type_as(x)
            
            length = self._max - self._min
            nbins = len(lut)
            
            idx = torch.floor((x - self._min) / length * nbins).long()
            idx[idx<0] = 0
            idx[idx>=nbins] = nbins-1
        
        return lut[idx]
    
    @classmethod
    def load(cls, filepath, key):
        with h5py.File(filepath, 'r') as f:
            grp = f[key]
            min = np.array(grp['min']).item()
            max = np.array(grp['max']).item()
            lut = np.array(grp['lut'])
        return cls(lut, min, max)

def get_weights(target):
    p1 = 6.72
    p2 = 2.33

    with torch.no_grad():
        weights = torch.exp(p1 * target + p2 * target**2)
        weights *= np.exp(p1 - p2)
        weights[target == -1] *= 0.1

    return weights

def get_grad_weights(target):
    gamma = 2.785
    x0 = 2.5

    with torch.no_grad():
        mask = target.abs() < x0
        weights = torch.empty_like(target)
        weights[mask] = torch.exp(gamma * target[mask].abs())
        weights[~mask] = np.exp(gamma * x0)

    return weights

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

class Siren(LightningModule):
    def __init__(self, cfg, name='siren', plib='photonlib'):
        super().__init__()

        model_cfg = cfg[name]
        self._lambda0 = model_cfg.get('lambda0', 1.)
        self._lambda1 = model_cfg.get('lambda1', 0.)

        net_pars = model_cfg['network']
        self.net = SirenNet(**net_pars)

        #self._lut = {
        #    key: WeightLUT.load(model_cfg['lut'], key)
        #    for key in ['vis']
        #}

        if model_cfg.get('lut') is not None:
            self._lut = PhotonLib.load(model_cfg['lut'], lib=torch)

        self.plib_cfg = cfg[plib]
        plib_fpath = self.plib_cfg['filepath']
        self.meta = Meta.load(plib_fpath, lib=torch)

    def forward(self, x):
        return self.net(x)
    
    def weighted_mse_loss(self, input, target, weight=1.):
        loss = (weight * (input - target)**2).mean()
        return loss

    def training_step_pair(self, batch, batch_id):
        x = batch['coord']
        target = batch['vis']
        weights = self._lut['vis'][target]

        pred, coord = self(x)
        loss = self.weighted_mse_loss(pred, target, weights)

        self.log('loss', loss)
        return loss

    def training_step_sumw(self, batch, batch_id):
        x = self.meta.voxel_to_coord(batch['idx'], norm=True)
        pred, coord = self(x)

        # ----------------------------------------------------------
        # calcuate gradient using sobel filter on 3x3 patches
        # ----------------------------------------------------------
        # sobel.shape = (3, 27), 3 gradidents and 3x3x3 cube
        # batch['vis_3x3'].shape = (batch_size, 27, n_pmts)
        # img_gt.shape = (27, batch_size * n_pmts)
        # grad_shape = (3, batch_size * n_pmts)
        # ----------------------------------------------------------
        sobel = get_sobel_coeffs().type_as(x)
        img_gt = torch.swapaxes(batch['vis_3x3'], 0, 1).reshape(27,-1)
        grad_gt = (sobel @ img_gt)
        max_grad = grad_gt.abs().max(axis=0).values
        weights = self._lut['grad'][max_grad]

        target = batch['vis'].flatten()
        weights += self._lut['vis'][target]

        loss = self.weighted_mse_loss(pred.flatten(), target, weights)

        self.log('loss', loss)

        return loss

    def training_step(self, batch, batch_id):
        voxel_id = batch['voxel_id']
        target = batch['vis']

        x = self.meta.voxel_to_coord(voxel_id, norm=True)
        pred, coord = self(x)

        target_for_lut = target
        #if not self.plib_cfg.get('transform', False):
        #    eps = self.plib_cfg.get('eps', 1e-7)
        #    target_for_lut = PhotonLib.transform(target, eps=eps, lib=torch)


        n_pmts = target.shape[-1]
        i0 = self.meta.voxel_to_idx(voxel_id)[:,0]
        i0 = i0.repeat(n_pmts, 1).swapaxes(0,1)

        i1 = torch.zeros_like(i0)
        i1[:,n_pmts//2:] = 1

        i2 = self._lut.meta.digitize(target_for_lut, 2)

        lut_idx = self._lut.meta.idx_to_voxel(
            torch.stack([i0,i1,i2], axis=-1).reshape(-1, 3)
        )
        lut = torch.tensor(self._lut.vis).type_as(x)
        weight = lut[lut_idx].reshape_as(target)

        #weight = self._lut['vis'][target_for_lut]
        
        #mask = target == 0
        #loss = 0.422 * self.weighted_mse_loss(pred[mask], target[mask])
        #loss += 0.578 * self.weighted_mse_loss(
        #    pred[~mask], target[~mask], weight[~mask]
        #)
        loss = self.weighted_mse_loss(pred, target, weight)
        
        self.log('loss', loss, on_step=False, on_epoch=True)

        eps = self.plib_cfg.get('eps', 1e-7)
        mask = target > PhotonLib.transform(4.5e-5, eps)
        if torch.any(mask):
            a = PhotonLib.inv_transform(pred[mask], eps, lib=torch)
            b = PhotonLib.inv_transform(target[mask], eps, lib=torch)
            bias = 2 * torch.abs(a - b) / (a + b)
            self.log('bias', bias.mean() * 100, prog_bar=True, on_step=False, on_epoch=True)


        #output = dict(batch)
        #output.update({
        #    'weight' : weight,
        #    'pred' : pred,
        #    'target' : target,
        #    'lut_idx' : lut_idx
        #})
        #torch.save(output, f'debug_{batch_id}.pkl')

        return loss

    def training_step_grad(self, batch, batch_idx):
        mask = batch['mask']

        #voxel_ids = batch['voxel_ids'][mask]
        #x = self.meta.voxel_to_coord(voxel_ids.flatten(), norm=True)
        #pred, coords = self(x)

        #target = batch['vis'][mask]
        #weights = self._lut['vis'][target]
        #loss_0 = self.weighted_mse_loss(pred, target, weights)

        voxel_ids = batch['voxel_ids']
        x = self.meta.voxel_to_coord(voxel_ids.flatten(), norm=True)
        pred, coords = self(x)

        target = batch['vis'][mask]
        pred_center = pred.reshape_as(batch['vis'])[mask]
        #weights = self._lut['vis'][target]

        loss_0 = self.weighted_mse_loss(pred_center, target)

        self.log('loss_0', loss_0)

        # ----------------------------------------------------------
        # gradient for ground truth
        # ----------------------------------------------------------
        # sobel.shape = (3, 27), 3 gradidents and 3x3x3 cube
        # sbatch['vis'].shape = (batch_size, 27, n_pmts)
        # img_gt.shape = (27, batch_size * n_pmts)
        # grad_shape = (3, batch_size * n_pmts)
        # ----------------------------------------------------------
        sobel = get_sobel_coeffs().type_as(x)

        img_gt = torch.swapaxes(batch['vis'], 0, 1).reshape(27,-1)
        grad_gt = (sobel @ img_gt).flatten()
        #grad_gt = (sobel @ img_gt).reshape(3, len(x), -1)
        #step_size = torch.as_tensor(self.meta.norm_step_size).type_as(x)
        #grad_gt /= 32 * step_size[:,None,None]

        # ----------------------------------------------------------
        # gradient for prediction
        # ----------------------------------------------------------
        # pred.shape = (batch_size * 27, n_pmts) 
        # img_pred.shape = (3, batch_size * n_pmts)
        # ----------------------------------------------------------
        img_pred = pred.reshape_as(batch['vis'])
        img_pred = torch.swapaxes(img_pred, 0, 1).reshape(27,-1)
        grad_pred = (sobel @ img_pred).flatten()


        #weights = self._lut['grad'][grad_gt.abs()]
        loss_1 = self.weighted_mse_loss(grad_pred, grad_gt)

        #loss_1 = 0
        #for pmt in range(pred.shape[1]):
        #    grad_pred = gradient(pred[:,pmt], coords)
        #    target = grad_gt[:,:,pmt].T
        #    weights = self._lut['grad'][target.abs()]
        #    loss_1 += self.weighted_mse_loss(grad_pred, target, weights)
        #loss_1 /= pred.shape[1]

        self.log('loss_1', loss_1)

        #loss = self._lambda0 * loss_0 + self._lambda1 * loss_1
        loss = loss_0 + loss_1
        self.log('loss', loss)

        return loss

    def training_step_part(self, batch, batch_idx):

        voxel_id = batch['voxel_id'].squeeze()
        x = self.meta.voxel_to_coord(voxel_id, norm=True)

        to_numpy = lambda t : t.squeeze().to('cpu', copy=True).numpy()

        shape = tuple(to_numpy(batch['shape']))

        padding = to_numpy(batch['padding'])
        padding[:,1] += shape[:-1]
        sx = slice(*padding[0])
        sy = slice(*padding[1])
        sz = slice(*padding[2])

        pred, coord = self(x)

        # --------------------------------
        target = batch['vis'].squeeze()
        target = target.reshape(shape)
        target = target[sx,sy,sz].flatten()

        pred_ = pred.reshape(shape)
        pred_ = pred_[sx,sy,sz].flatten()

        #weights = self._lut['vis'][target]
        weights = get_weights(target)
        loss_0 = self.weighted_mse_loss(pred_, target, weights)

        self.log('loss_0', loss_0)

        # --------------------------------
        img_pred = pred.reshape(shape)
        img_pred = torch.moveaxis(img_pred.unsqueeze(0), -1, 0)
        grad_pred = self._sobel(img_pred).squeeze()
        grad_pred = grad_pred[:,sx,sy,sz]

        img_gt = batch['vis'].squeeze().reshape(shape)
        img_gt = torch.moveaxis(img_gt.unsqueeze(0), -1, 0)
        grad_gt = self._sobel(img_gt).squeeze()
        grad_gt = grad_gt[:,sx,sy,sz]

        loss_1 = self.weighted_mse_loss(grad_pred, grad_gt)


        self.log('loss_1', loss_1)

        # --------------------------------
        loss = self._lambda0 * loss_0 + self._lambda1 * loss_1
        self.log('loss', loss)

        return loss

    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr=1e-5)
        opt = torch.optim.Adam(self.parameters(), lr=1e-7)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5, threshold=1e-4, 
            threshold_mode='rel', cooldown=1, verbose=True)

        return opt
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler' : sch,
                'interval'  : 'epoch',
                'monitor'   : 'loss',
            },
        }
