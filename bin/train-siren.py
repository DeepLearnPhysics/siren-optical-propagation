#!/usr/bin/env python3

import yaml
import torch
import fire
import os
import time
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

#from siren import Siren 
#from siren.io import dataloader_factory, PhotonLibWrapper
from siren.io import dataloader_factory
from siren.utils import import_from

def train( 
    cfg_file, lr=None, load=None, resume=None, 
    max_epochs=10000, gpus=1, uuid=False, log=None,
):

    # check input arguments
    if load is not None and resume is not None:
        print('[Error] --load and --resume cannot be used together',
              file=sys.stderr)
        return

    # prepare config dict
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    if lr is not None:
        cfg['siren']['lr'] = lr

    # -------------------------------------------------------------------------
    # dataloader
    # -------------------------------------------------------------------------
    dataloader = dataloader_factory(cfg)

    # -------------------------------------------------------------------------
    # logger
    # -------------------------------------------------------------------------
    logger_cfg = cfg.get('logger', {})

    if log is None:
        log_dir = logger_cfg.get('log_dir', 'logs')
        log_name = logger_cfg.get('name', None)
    else:
        log_dir, log_name = os.path.split(log)

    if log_name is None:
        log_name = 'siren'
        uuid = True

    if uuid:
        pid = os.getpid()
        ts = int(time.time())
        log_name = f'{log_name}_{ts:x}_{pid}'

    logger = CSVLogger(log_dir, name=log_name)

    Model = import_from(cfg['class']['model'])

    if load is None:
        model = Model(cfg)
    else:
        print(f'[INFO] load {load}')
        model = Model.load_from_checkpoint(load, strict=False, cfg=cfg)

    train_cfg = cfg.get('training', {})
    ckpt_callback = ModelCheckpoint(**train_cfg.get('checkpoint', {}))

    trainer_cfg = train_cfg.get('trainer', {})
    trainer_cfg.update(dict(
        gpus=gpus,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[ckpt_callback],
    ))

    trainer = Trainer(**trainer_cfg)

    if resume is None:
        trainer.fit(model, dataloader)
    else:
        print(f'[INFO] resume {resume}')
        trainer.fit(model, dataloader, ckpt_path=resume)

if __name__ == '__main__':
    fire.Fire(train)
