"""
@file main_stationary.py

Main entrypoint for training the stationary models over a set of tasks.
"""
import torch
import hydra
import random
import pytorch_lightning
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.dataloader_ep import StationaryEPDataModule
from utils.utils import get_model, flatten_cfg
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


@hydra.main(version_base="1.3", config_path="configs", config_name="stationary")
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

    # Build datasets based on tasks
    dataset = StationaryEPDataModule(cfg, task_ids=cfg.task_ids)
    
    # Initialize model
    model = get_model(cfg.model)(cfg)

    # Set up parameters for each patient
    for data_idx, data_name in enumerate(cfg.data_names):
        model.construct_nodes(data_idx, data_name, 'data/ep/', cfg.batch_size, cfg.devices[0], cfg.load_torso, cfg.load_physics, cfg.graph_method)

    # Set up the logger if its train
    logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}")

    # Defining the Trainer
    trainer = pytorch_lightning.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=0,
        max_steps=0,
        gradient_clip_val=cfg.gradient_clip,
        val_check_interval=cfg.val_log_interval,
        num_sanity_val_steps=0
    )
    trainer.callbacks.append(None)

    # Callbacks for logging and tensorboard
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{logger.log_dir}/checkpoints/',
        monitor='train_scc',
        mode='max',
        filename='step{step:02d}-scc{train_scc:.4f}',
        every_n_train_steps=cfg.log_interval,
        auto_insert_metric_name=False,
        save_last=True
    )

    # Extend training by another iteration
    trainer.callbacks[-2] = checkpoint_callback
    trainer.callbacks[-1] = lr_monitor
    trainer.logger = logger
    trainer.fit_loop.max_epochs += 1
    trainer.fit_loop.max_steps += cfg.num_steps * cfg.batch_size

    # Training the model
    trainer.fit(model, dataset.train_dataloader())

    # Set up parameters for each patient
    for data_idx, data_name in enumerate(cfg.data_names):
        model.construct_nodes(data_idx, data_name, 'data/ep/', 1, cfg.devices[0], cfg.load_torso, cfg.load_physics, cfg.graph_method)

    # Test on the training set
    for task_id in cfg.task_ids:
        cfg.split = f"{task_id}_train"
        trainer.test(model, dataset.test_dataloader(task_id=task_id, mode="train"), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")

    # Test on the training set
    for task_id in cfg.task_ids:
        cfg.split = str(task_id)
        trainer.test(model, dataset.test_dataloader(task_id=task_id, mode="val"), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")


if __name__ == '__main__':
    main()