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

from siren import Siren 
from siren.io import dataloader_factory, PhotonLibWrapper

def train(cfg_file, lr=None, load=None, resume=None, max_epochs=10000):

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

    # dataloader
    dataloader = dataloader_factory(PhotonLibWrapper, cfg)

    # logger
    logger_cfg = cfg.get('logger', {})
    log_dir = logger_cfg.get('log_dir', 'logs')
    log_name = logger_cfg.get('name', None)
    if log_name is None:
        pid = os.getpid()
        ts = int(time.time())
        log_name = f'siren_{ts:x}_{pid}'

    logger = CSVLogger(log_dir, name=log_name)

    if load is None:
        model = Siren(cfg)
    else:
        print(f'[INFO] load {load}')
        model = Siren.load_from_checkpoint(load, cfg=cfg)

    ckpt_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='bias',
        mode='min',
        every_n_epochs=1,
    )

    # trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=max_epochs,
        log_every_n_steps=8,
        logger=logger,
        callbacks=[ckpt_callback],
    )

    if resume is None:
        trainer.fit(model, dataloader)
    else:
        print(f'[INFO] resume {resume}')
        trainer.fit(model, dataloader, ckpt_path=resume)

if __name__ == '__main__':
    fire.Fire(train)
