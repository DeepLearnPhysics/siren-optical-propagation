import torch
import numpy as np

from siren.base import Siren
from photonlib import PhotonLib

class SirenVis(Siren):
    def __init__(self, cfg, name='siren'):
        siren_cfg = cfg[name]
        
        # initialize Siren class
        net_cfg = siren_cfg['network']
        super().__init__(**net_cfg)
        
        # extensions for visiblity model
        self._init_output_scale(siren_cfg)
        self._do_hardsigmoid = siren_cfg.get('hardsigmoid', False)
        
        # tranform visiblity in pseudo-log scale (default: False)
        plib_cfg = cfg.get('photonlib', {})
        self.is_log_scale = plib_cfg.get('transform', False)
        if self.is_log_scale:
            self.transform, self.inv_transform = PhotonLib.partial_transform(
                vmax=plib_cfg.get('vmax', 1),
                eps=plib_cfg.get('eps', 1e-7),
            )
        else:
            identity = lambda x, lib=None: x
            self.transform, self.inv_transform = identity, identity
            
    def _init_output_scale(self, siren_cfg):
        n_outs = siren_cfg['network']['out_features']

        scale_cfg = siren_cfg.get('output_scale', {})
        init = scale_cfg.get('init')
        
        # 1) set scale=1 (default)
        if init is None:
            scale = np.ones(n_outs)
            
        # 2) load from np file
        elif isinstance(init, str):
            scale = np.load(init)
        
        # 3) take from cfg as-it
        else:
            scale = np.asarray(init)
            
        assert len(scale)==n_outs, 'len(output_scale) != out_features'
        
        scale = torch.tensor(np.nan_to_num(scale), dtype=torch.float32)
        
        if scale_cfg.get('fix', True):
            self.register_buffer('scale', scale)
        else:
            self.register_parameter('scale', torch.nn.Parameter(scale))
            
    def forward(self, x, to_vis=False):
        out, coords = super().forward(x)
        
        if self._do_hardsigmoid:
            out =  torch.nn.functional.hardsigmoid(out)
            
        if to_vis and self.is_log_scale:
            out = self.inv_transform(out, lib=torch)
            
        out = out * self.scale
        
        return out, coords
