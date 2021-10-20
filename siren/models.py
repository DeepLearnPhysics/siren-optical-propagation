import torch
import numpy as np
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule

from siren.networks import Siren as SirenNet
from photonlib import PhotonLib, Meta

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

class Siren(LightningModule):
    def __init__(self, cfg, name='siren', plib='photonlib'):
        super().__init__()

        pars = cfg[name]
        self.net = SirenNet(**pars)

        plib_fpath = cfg[plib]['filepath']
        self.meta = Meta.load_file(plib_fpath, lib=torch)

        self._max_vis = MaxMetric()

        self._weights = torch.load(
            '/sdf/group/neutrino/kvtsang/ml/opreco/plib_data/event_weights.pk')
    
    def forward(self, x):
        return self.net(x)
    
    def weighted_mse_loss(self, input, target, weights):
        loss = (weights * (input - target)**2).mean()
        return loss

    def get_weights(self, target):
        #gamma = 2.247
        #min_weight = 1e-7
        #threshold = np.exp(np.log(min_weight) / gamma)

        #mask = target > threshold
        #weights = torch.empty_like(target)
        #weights[~mask] = min_weight
        #weights[mask] = torch.pow(target[mask], gamma)
        #weights[target == 0] = min_weight * 0.1
        #weights /= min_weight * 0.1;

        p1 = 6.72
        p2 = 2.33
        weights = torch.exp(p1 * target + p2 * target**2)
        weights *= np.exp(p1 - p2)
        weights[target == -1] *= 0.1
        return weights

        #w = torch.as_tensor(self._weights, device=target.device)
        #n = len(w)
        #idx = (target * n).long()
        #idx[idx == n] = n - 1
        #return w[idx]

    def training_step(self, batch, batch_idx):

        voxel_id = batch['voxel_id']
        target = self.meta.transform(batch['vis'])
        weights = self.get_weights(target)

        x = self.meta.voxel_to_coord(voxel_id, norm=True)
        pred, __ = self(x)
        loss = self.weighted_mse_loss(pred, target, weights)

        self.log('loss', loss)
        self._max_vis(batch['vis'].max())

        #torch.save(batch, f'batch_{batch_idx}.pk')

        return loss

    def training_step_end(self, outputs):
        self.log('max_vis', self._max_vis.compute())
        self._max_vis.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-5)
        #sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95, verbose=True)
        #sch = torch.optim.lr_scheduler.LambdaLR(opt, self._lr_lambda, verbose=True)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=10, threshold=1e-4, 
            threshold_mode='rel', cooldown=10, verbose=True)

        #sch = torch.optim.lr_scheduler.MultiStepLR(
        #    opt, milestones=[1500, 5000], gamma=0.1)

        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler' : sch,
                'interval'  : 'epoch',
                'monitor'   : 'loss',
            },
        }
