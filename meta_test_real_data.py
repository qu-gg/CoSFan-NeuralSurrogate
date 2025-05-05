"""
@file main_continual.py

Main entrypoint for training the CL methods on a random sequence of tasks.
"""
import os
import json
import torch
import hydra
import random
import numpy as np
import pytorch_lightning
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from sklearn.manifold import TSNE
from utils.utils import get_model, flatten_cfg
from utils.dataloader_ep import ContinualEPDataModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@hydra.main(version_base="1.3", config_path="configs", config_name="continual")
def main(cfg: DictConfig):
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(cfg.seed, workers=True)
    random.seed(123123)

    # Disable logging for true runs
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)
    
    # Enable fp16 training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    # Limit number of CPU workers
    torch.set_num_threads(8)

    # Flatten the Hydra config
    cfg.exptype = cfg.exptype
    cfg = flatten_cfg(cfg)
    
    # If we're testing MAML models, set LR to 0 automatically
    if cfg.train is not True and cfg.model == "maml":
        cfg.learning_rate = 0

    print(f"=> Sequence of Tasks: {cfg.task_ids}")

    # Build datasets based on tasks
    datamodules = dict()
    for task_id in cfg.task_ids:
        datamodules[task_id] = ContinualEPDataModule(cfg, [task_id])
        print(f"=> Task {task_id}")
        print(f"=> Dataset 'test' xs shape: {datamodules[task_id].test_dataloader().dataset.xs.shape}")
        print(f"=> Dataset 'test' labels shape: {datamodules[task_id].test_dataloader().dataset.labels.shape}")

    # Initialize model
    model = get_model(cfg.model)(cfg)
    
    # Set up parameters for each patient
    for data_idx, data_name in enumerate(cfg.data_names):
        model.construct_nodes(data_idx, data_name, 'data/ep/', cfg.batch_size, cfg.devices[0], cfg.load_torso, cfg.load_physics, cfg.graph_method)

    # Set up the logger if its train
    logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}", version=cfg.version)

    # Defining the Trainer
    trainer = pytorch_lightning.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=0,
        max_steps=0,
        gradient_clip_val=cfg.gradient_clip,
        # val_check_interval=cfg.val_log_interval,
        val_check_interval=None,
        num_sanity_val_steps=0,
    )
    trainer.callbacks.append(None)

    # Iterate over tasks, defining the new Task Trainer and evaluating after training
    for idx, task_id in enumerate(cfg.task_ids):
        # Callbacks for logging and tensorboard
        task_logger = pl_loggers.TensorBoardLogger(save_dir=logger.log_dir, name=f"task_{idx}", version='')
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{task_logger.log_dir}/checkpoints/',
            monitor='val_likelihood',
            filename='step{step:02d}-val_likelihood{val_likelihood:.2f}',
            auto_insert_metric_name=False,
            save_last=True
        )

        # Extend training by another iteration
        trainer.callbacks[-2] = checkpoint_callback
        trainer.callbacks[-1] = lr_monitor
        trainer.logger = task_logger
        trainer.fit_loop.max_epochs += 1
        trainer.fit_loop.max_steps += cfg.num_task_steps * cfg.batch_size

        if cfg.model == "maml":
            trainer.fit_loop.max_steps = 1
            trainer.fit(model, datamodules[task_id])

        # Test on all sets
        for task in cfg.task_ids:
            cfg.split = f"{task}_heart"
            trainer.test(model, datamodules[task], ckpt_path=f"{task_logger.log_dir}/checkpoints/last.ckpt")


if __name__ == '__main__':
    main()
