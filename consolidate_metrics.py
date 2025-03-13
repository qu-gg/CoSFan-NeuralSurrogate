"""
@file main_stationary.py

Main entrypoint for training the stationary models over a set of tasks.
"""
import hydra
import json
import numpy as np
import pytorch_lightning.loggers as pl_loggers

from omegaconf import DictConfig
from utils.utils import flatten_cfg


@hydra.main(version_base="1.3", config_path="configs", config_name="stationary")
def main(cfg: DictConfig):
    # Flatten the Hydra config
    cfg.exptype = cfg.exptype
    cfg = flatten_cfg(cfg)

    # Set up the logger if its train
    logger = pl_loggers.TensorBoardLogger(save_dir=f"experiments/{cfg.exptype}/", name=f"{cfg.model}", version=6)
    print(logger.log_dir)

    train_metrics = {
        'mse_mean': [],
        'mse_std': [],
        'scc_mean': [],
        'scc_std': [],
        'tcc_mean': [],
        'tcc_std': []
    }
    
    test_metrics = {
        'mse_mean': [],
        'mse_std': [],
        'scc_mean': [],
        'scc_std': [],
        'tcc_mean': [],
        'tcc_std': []
    }
    for task_id in cfg.task_ids:
        train_json = json.load(open(f"{logger.log_dir}/test_{task_id}_train/test_{task_id}_train_metrics.json", 'r'))
        for key in train_json.keys():
            train_metrics[key].append(train_json[key])
        
        test_json = json.load(open(f"{logger.log_dir}/test_{task_id}/test_{task_id}_metrics.json", 'r'))
        for key in test_json.keys():
            test_metrics[key].append(test_json[key])
    
    print(train_metrics)
    
    for key in train_metrics.keys():
        train_metrics[key] = np.mean(train_metrics[key])

    for key in test_metrics.keys():
            test_metrics[key] = np.mean(test_metrics[key])
            
    print(train_metrics)
    print(test_metrics)
    
    with open(f"{logger.log_dir}/metrics_train.json", 'w') as f:
        json.dump(train_metrics, f)
        
    with open(f"{logger.log_dir}/metrics_test.json", 'w') as f:
        json.dump(test_metrics, f)

if __name__ == '__main__':
    main()