"""
@file test_meta_adaptation.py

Script to handle benchmarking the time it takes to perform the 
meta-adaptation of the MAML-based and FeedForward-based models
"""
import torch
import numpy as np
import time
from omegaconf import OmegaConf
import hydra

from tqdm import tqdm
from collections import defaultdict

from models.FeedForward import FeedForward
from utils.utils import flatten_cfg
from models.MAML import Maml


def create_synthetic_domain_data(mesh_size, domain_size=5, timesteps=55, batch_size=8):
    """Create synthetic domain data for a single task"""
    # Create domain signals (N, K, V, T)
    x = torch.randn(1, mesh_size, timesteps)
    x_domain = torch.randn(1, domain_size, mesh_size, timesteps)
    
    # Create domain labels (N, K, 3) - assuming 3D labels
    y = torch.ones(1, 3).long()
    y_domain = torch.ones(1, domain_size, 3).long()
    
    return x, x_domain, y, y_domain


def setup_model(model_type, cfg):
    """Initialize a model based on type"""
    if model_type == "feedforward":
        return FeedForward(cfg)
    elif model_type == "maml":
        return Maml(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def benchmark_latent_domain(model, model_name, x, x_domain, y, y_domain, heart_name, device, num_trials=100, num_tasks=1):
    """Measure latent_domain computation time"""
    times = []
    
    # Move data to device
    x = x.to(device)
    x_domain = x_domain.to(device)
    
    y = y.to(device)
    y_domain = y_domain.to(device)
    
    # Warmup runs
    for _ in tqdm(range(5)):
        if model_name == "feedforward":
            _ = model.latent_domain(x_domain, y_domain, heart_name)
        elif model_name == "maml":
            _ = model.single_fast_weight(x, x_domain, y, y_domain, heart_name)
    
    # Actual timing runs
    for _ in tqdm(range(num_trials)):
        start_time = time.time()
        for _ in range(num_tasks):
            if model_name == "feedforward":
                _ = model.latent_domain(x_domain, y_domain, heart_name)
            elif model_name == "maml":
                _ = model.single_fast_weight(x, x_domain, y, y_domain, heart_name)

        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def benchmark_forward(model, model_name, x, x_domain, y, y_domain, heart_name, device, num_trials=100):
    """Measure latent_domain computation time"""
    times = []
    
    # Move data to device
    x = x.to(device)
    x_domain = x_domain.to(device)
    
    y = y.to(device)
    y_domain = y_domain.to(device)
    
    # Warmup runs
    for _ in range(5):
        if model_name == "feedforward":
            _ = model(x, x_domain, y, y_domain, heart_name)
        elif model_name == "maml":
            _ = model(x, x_domain, y, y_domain, heart_name)

    # Actual timing runs
    for _ in range(num_trials):
        start_time = time.time()
        
        if model_name == "feedforward":
            _ = model(x, x_domain, y, y_domain, heart_name)
        elif model_name == "maml":
            _ = model(x, x_domain, y, y_domain, heart_name)

        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


@hydra.main(version_base='1.3', config_path="configs", config_name="ablation")
def main(cfg):
    # Flatten the Hydra config
    cfg.exptype = cfg.exptype
    cfg = flatten_cfg(cfg)
    model_name = "feedforward"
    
    cfg.devices = [2]
    
    # Set device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # Mesh configurations
    mesh_sizes = {480: "AW", 475: "DC", 448: "EC"}
    model = setup_model(model_name, cfg)
    model = model.to(device)
    model.train()
    
    # Create synthetic domain data
    x, x_domain, y, y_domain = create_synthetic_domain_data(
        mesh_size=480,
        domain_size=cfg.domain_size,
        timesteps=55,
        batch_size=cfg.batch_size
    )
    
    # Initialize task-specific graph structures if needed
    model.construct_nodes(
        data_idx=0,
        heart_name="AW",
        data_path='data/ep/',
        batch_size=1,
        k_shot=cfg.domain_size,
        device=device,
        load_torso=cfg.load_torso,
        load_physics=cfg.load_physics,
        graph_method=cfg.graph_method
    )
    
    # # Benchmark latent_domain function
    # mean_time, std_time = benchmark_latent_domain(model, model_name, x, x_domain, y, y_domain, 0, device, num_trials=100, num_tasks=1)
    # print(f"Average latent_domain time [1 task]: {mean_time:.4f} ± {std_time:.4f} seconds")
    
    # # Benchmark latent_domain function
    # mean_time, std_time = benchmark_latent_domain(model, model_name, x, x_domain, y, y_domain, 0, device, num_trials=100, num_tasks=3)
    # print(f"Average latent_domain time [9 tasks]: {mean_time:.4f} ± {std_time:.4f} seconds")
    
    # Benchmark forward function
    mean_time, std_time = benchmark_forward(model, model_name, x, x_domain, y, y_domain, 0, device, num_trials=100)
    print(f"Average forward time [1 task]: {mean_time:.4f} ± {std_time:.4f} seconds")
    

if __name__ == "__main__":
    main()
