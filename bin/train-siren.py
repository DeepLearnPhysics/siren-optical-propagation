#!/usr/bin/env python3

import yaml
import torch
import fire
import os
import time

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from siren import Siren 
from siren.io import dataloader_factory, PhotonLibWrapper

def train(cfg_file, lr=None, checkpoint=None, max_epochs=10000):

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

    # checkpoint
    if checkpoint is None:
        model = Siren(cfg)
    else:
        model = Siren.load_from_checkpoint(checkpoint, cfg=cfg)

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
        log_every_n_steps=37,
        logger=logger,
        callbacks=[ckpt_callback],
    )

    trainer.fit(model, dataloader)

if __name__ == '__main__':
    fire.Fire(train)
