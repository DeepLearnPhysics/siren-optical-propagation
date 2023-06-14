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
    max_epochs=10000, gpus=1, uid=False, log=None,
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

    # dataloader
    dataloader = dataloader_factory(cfg)
    val_dataloader = None
    if 'val_dataloader' in cfg:
        val_dataloader = dataloader_factory(
            cfg, data_key='val_dataset', loader_key='val_dataloader'
        )

    # uid
    pid = os.getpid()
    ts = int(time.time())

    # -------------------------------------------------------------------------
    # logger
    # -------------------------------------------------------------------------
    logger_cfg = cfg.setdefault('logger', {})

    if log is None:
        log_dir = logger_cfg.get('log_dir', 'logs')
        log_name = logger_cfg.get('name', None)
    else:
        log_dir, log_name = os.path.split(log)

    if log_name is None:
        log_name = 'siren'
        uid = True

    if uid:
        log_name = f'{log_name}_{ts:x}_{pid}'

    logger = CSVLogger(log_dir, name=log_name)
    logger_cfg['log_dir'] = log_dir
    logger_cfg['name'] = log_name
    
    # -------------------------------------------------------------------------
    # model
    # -------------------------------------------------------------------------
    Model = import_from(cfg['class']['model'])
    train_cfg = cfg.setdefault('training', {})
    train_cfg['runtime'] = {'max_epochs': max_epochs}
    runtime_cfg = train_cfg['runtime']

    if load is None:
        model = Model(cfg)
    else:
        print(f'[INFO] load {load}')
        model = Model.load_from_checkpoint(load, strict=False, cfg=cfg)
        runtime_cfg['load'] = load

    # -------------------------------------------------------------------------
    # Trainer + Checkpoint
    # -------------------------------------------------------------------------
    ckpt_callback = ModelCheckpoint(**train_cfg.get('checkpoint', {}))

    trainer_kwargs = train_cfg.get('trainer', {}).copy()
    trainer_kwargs.update(dict(
        gpus=gpus,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[ckpt_callback],
    ))

    trainer = Trainer(**trainer_kwargs)
    if resume is not None:
        runtime_cfg['resume'] = resume

    # -------------------------------------------------------------------------
    # save cfg file
    # -------------------------------------------------------------------------
    cfg_dir = os.path.join(log_dir, log_name, 'cfg')
    os.makedirs(cfg_dir)
    with open(f'{cfg_dir}/siren_{ts:x}_{pid}.yaml', 'w') as f:
        yaml.safe_dump(cfg, f)


    # -------------------------------------------------------------------------
    # start training
    # -------------------------------------------------------------------------
    if resume is None:
        trainer.fit(model, dataloader, val_dataloader)
    else:
        print(f'[INFO] resume {resume}')
        trainer.fit(model, dataloader, val_dataloader, ckpt_path=resume)

if __name__ == '__main__':
    fire.Fire(train)
