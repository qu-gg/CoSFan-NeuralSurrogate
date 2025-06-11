"""
@file main_stationary.py

Main entrypoint for training the stationary models over a set of tasks.
"""
import json
import torch
import hydra
import random
import numpy as np
import pytorch_lightning
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.dataloader_ep import StationaryEPDataModule
from utils.utils import get_model, flatten_cfg
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def consolidate_metrics(cfg, log_dir, mode=""):
    """ Consolidates all the metrics over the tasks into one file """
    import glob
    import json
    import numpy as np
    
    # Find all generalization test metric files
    train_files = glob.glob(f"{log_dir}/test_*{mode}/test_*{mode}_metrics.json")
    
    # Initialize metric lists
    mses, sccs, dccs = [], [], []
    
    # Extract metrics from each file
    for f in train_files:
        try:
            with open(f, 'r') as file:
                metrics = json.load(file)
                mses.append(metrics['mse_mean'])
                sccs.append(metrics['scc_mean'])
                dccs.append(metrics['dcc_mean'])
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    # Calculate summary statistics
    metrics = {
        'mse_mean': np.nanmean(mses),
        'mse_std': np.nanstd(mses),
        'scc_mean': np.nanmean(sccs),
        'scc_std': np.nanstd(sccs),
        'dcc_mean': np.nanmean(dccs),
        'dcc_std': np.nanstd(dccs)
    }
    
    # Save metrics in Excel-friendly format
    with open(f"{log_dir}/metrics_excel{mode}.txt", 'w') as f:
        f.write(f"MSE:, {metrics['mse_mean']:0.5f}({metrics['mse_std']:0.5f})\n")
        f.write(f"SCC:, {metrics['scc_mean']:0.5f}({metrics['scc_std']:0.5f})\n")
        f.write(f"DCC:, {metrics['dcc_mean']:0.5f}({metrics['dcc_std']:0.5f})")
    
    return metrics


def consolidate_generalization_metrics(log_dir):
    """Consolidates generalization metrics from multiple test files into a single summary.
    
    Args:
        log_dir (str): Path to the directory containing test results
    """
    import glob
    import json
    import numpy as np
    
    # Find all generalization test metric files
    train_files = glob.glob(f"{log_dir}/test_generalization_*/test_generalization_*_metrics.json")
    
    # Initialize metric lists
    mses, sccs, dccs = [], [], []
    
    # Extract metrics from each file
    for f in train_files:
        try:
            with open(f, 'r') as file:
                metrics = json.load(file)
                mses.append(metrics['mse_mean'])
                sccs.append(metrics['scc_mean'])
                dccs.append(metrics['dcc_mean'])
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    
    # Calculate summary statistics
    metrics = {
        'mse_mean': np.nanmean(mses),
        'mse_std': np.nanstd(mses),
        'scc_mean': np.nanmean(sccs),
        'scc_std': np.nanstd(sccs),
        'dcc_mean': np.nanmean(dccs),
        'dcc_std': np.nanstd(dccs)
    }
    
    # Save metrics in Excel-friendly format
    with open(f"{log_dir}/metrics_generalization_excel.txt", 'w') as f:
        f.write(f"MSE:, {metrics['mse_mean']:0.5f}({metrics['mse_std']:0.5f})\n")
        f.write(f"SCC:, {metrics['scc_mean']:0.5f}({metrics['scc_std']:0.5f})\n")
        f.write(f"DCC:, {metrics['dcc_mean']:0.5f}({metrics['dcc_std']:0.5f})")
    
    return metrics



@hydra.main(version_base="1.3", config_path="configs", config_name="stationary")
def main(cfg: DictConfig):
    # Set a consistent seed over the full set for consistent analysis
    pytorch_lightning.seed_everything(cfg.seed, workers=True)
    random.seed(cfg.seed)

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
        model.construct_nodes(data_idx, data_name, 'data/ep/', cfg.batch_size, cfg.domain_size, cfg.devices[0], cfg.load_torso, cfg.load_physics, cfg.graph_method)

    # Set up the logger if its train
    logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}", version=0)

    # Defining the Trainer
    trainer = pytorch_lightning.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=0,
        max_steps=0,
        gradient_clip_val=cfg.gradient_clip,
        val_check_interval=cfg.val_log_interval,
        num_sanity_val_steps=0,
        accumulate_grad_batches=5
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
    # trainer.fit(model, dataset.train_dataloader())

    # Test on the training set
    # for task_id in cfg.task_ids:
    #     cfg.split = f"{task_id}_train"
    #     trainer.test(model, dataset.evaluate_train_dataloader(task_id=task_id), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")

    # Test on the training set
    # for task_id in cfg.task_ids:
    #     cfg.split = f"{task_id}"
    #     trainer.test(model, dataset.test_dataloader(task_id=task_id), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")

    # Test on the training set
    for task_id in cfg.test_task_ids:
        cfg.split = f"generalization_{task_id}"
        trainer.test(model, dataset.test_dataloader(task_id=task_id), ckpt_path=f"{logger.log_dir}/checkpoints/last.ckpt")
 
    # Consolidate final metrics
    consolidate_metrics(cfg, f"{logger.log_dir}/", mode="")
    consolidate_metrics(cfg, f"{logger.log_dir}/", mode="_train")
    consolidate_generalization_metrics(f"{logger.log_dir}/")


if __name__ == '__main__':
    main()