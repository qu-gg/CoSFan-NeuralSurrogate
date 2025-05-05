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


def consolidate_metrics(cfg, logger, mode=""):
    """ Consolidates all the metrics over the tasks into one file """
    # Get the performance metrics across tasks
    task_performances = dict()
    for task_id in np.unique(cfg['task_ids']):
        task_performances[f"task_{task_id}"] = dict()
        for metric in cfg['test_metrics']:
            task_performances[f"task_{task_id}"][f'{metric}_mean'] = [np.nan for _ in range(len(cfg['task_ids']))]
            task_performances[f"task_{task_id}"][f'{metric}_std'] = [np.nan for _ in range(len(cfg['task_ids']))]

    for idx in range(len(cfg['task_ids'])):
        for task_idx in range(len(cfg['task_ids'][:idx + 1])):
            true_task_id = cfg['task_ids'][task_idx]

            try:
                task_metrics = json.load(
                    open(f"{logger.log_dir}/task_{idx}/test_{task_idx}{mode}/test_{task_idx}{mode}_metrics.json")
                )

                for metric in cfg['test_metrics']:
                    task_performances[f"task_{true_task_id}"][f'{metric}_mean'][idx] = task_metrics[f'{metric}_mean']
                    task_performances[f"task_{true_task_id}"][f'{metric}_std'][idx] = task_metrics[f'{metric}_std']

            except Exception as e:
                continue

    """ Get RP metrics """
    lp_mses, lp_scc, lp_tcc = [], [], []
    rp_mses, rp_scc, rp_tcc = [], [], []
    for idx, key in enumerate(cfg['task_ids']):
        lp_mses.append(task_performances[f"task_{key}"]['mse_mean'][idx])
        rp_mses.append(task_performances[f"task_{key}"]['mse_mean'][-1])

        lp_scc.append(task_performances[f"task_{key}"]['scc_mean'][idx])
        rp_scc.append(task_performances[f"task_{key}"]['scc_mean'][-1])

        lp_tcc.append(task_performances[f"task_{key}"]['tcc_mean'][idx])
        rp_tcc.append(task_performances[f"task_{key}"]['tcc_mean'][-1])

    bti_mses, bti_scc, bti_tcc = [], [], []
    for idx in range(len(lp_scc)):
        bti_mses.append(lp_mses[idx] - rp_mses[idx])
        bti_scc.append(rp_scc[idx] - lp_scc[idx])
        bti_tcc.append(rp_tcc[idx] - lp_tcc[idx])

    metrics = {
        'lp_mses_mean': np.mean(lp_mses),
        'lp_mses_std': np.std(lp_mses),
        'rp_mses_mean': np.mean(rp_mses),
        'rp_mses_std': np.std(rp_mses),
        'bti_mses_mean': np.mean(bti_mses),
        'bti_mses_std': np.std(bti_mses),
        
        'lp_scc_mean': np.mean(lp_scc),
        'lp_scc_std': np.std(lp_scc),
        'rp_scc_mean': np.mean(rp_scc),
        'rp_scc_std': np.std(rp_scc),
        'bti_scc_mean': np.mean(bti_scc),
        'bti_scc_std': np.std(bti_scc),
        
        'lp_tcc_mean': np.mean(lp_tcc),
        'lp_tcc_std': np.std(lp_tcc),
        'rp_tcc_mean': np.mean(rp_tcc),
        'rp_tcc_std': np.std(rp_tcc),
        'bti_tcc_mean': np.mean(bti_tcc),
        'bti_tcc_std': np.std(bti_tcc),
    }

    # Save metrics into an easy text-based file
    with open(f"{logger.log_dir}/metrics_excel.txt", 'w') as f:
        f.write(f"MSE:, {metrics['lp_mses_mean']:0.5f}({metrics['lp_mses_std']:0.5f}), {metrics['rp_mses_mean']:0.5f}({metrics['rp_mses_std']:0.5f}), {metrics['bti_mses_mean']:0.5f}({metrics['bti_mses_std']:0.5f})\n")
        f.write(f"SCC:, {metrics['lp_scc_mean']:0.5f}({metrics['lp_scc_std']:0.5f}), {metrics['rp_scc_mean']:0.5f}({metrics['rp_scc_std']:0.5f}), {metrics['bti_scc_mean']:0.5f}({metrics['bti_scc_std']:0.5f})\n")
        f.write(f"TCC:, {metrics['lp_tcc_mean']:0.5f}({metrics['lp_tcc_std']:0.5f}), {metrics['rp_tcc_mean']:0.5f}({metrics['rp_tcc_std']:0.5f}), {metrics['bti_tcc_mean']:0.5f}({metrics['bti_tcc_std']:0.5f})")

    # Save to a JSON in the logger directory
    with open(f"{logger.log_dir}/metrics{mode}.json", 'w') as f:
        json.dump(metrics, f)


def plot_continual_metrics(cfg, logger, mode=""):
    """ Handles plotting a continual performance plot of each unique dynamic over the task numbers for every metric """
    # Get the performance metrics across tasks
    task_performances = dict()
    for task_id in np.unique(cfg.task_ids):
        task_performances[f"task_{task_id}"] = dict()
        for metric in cfg['test_metrics']:
            task_performances[f"task_{task_id}"][f'{metric}_mean'] = [np.nan for _ in range(len(cfg['task_ids']))]
            task_performances[f"task_{task_id}"][f'{metric}_std'] = [np.nan for _ in range(len(cfg['task_ids']))]

    for idx in range(len(cfg.task_ids)):
        for task_idx in range(len(cfg.task_ids[:idx + 1])):
            true_task_id = cfg.task_ids[task_idx]

            try:
                task_metrics = json.load(
                    open(f"{logger.log_dir}/task_{idx}/test_{task_idx}{mode}/test_{task_idx}{mode}_metrics.json")
                )

                for metric in cfg['test_metrics']:
                    task_performances[f"task_{true_task_id}"][f'{metric}_mean'][idx] = task_metrics[f'{metric}_mean']
                    task_performances[f"task_{true_task_id}"][f'{metric}_std'][idx] = task_metrics[f'{metric}_std']

            except Exception as e:
                continue

    def plot_metric(metric_name):
        """ Handles plotting single metric plot """
        plt.rcParams['figure.figsize'] = (10, 5)
        fig, ax = plt.subplots()

        # Plot the performances over tasks over time
        markers = ['o', 'v', '^', '<', '>', 's', '8', 'p', 'o', 'v', '^', '<', '>', 's', '8', 'p', 's', '8', 'p']
        dynamics_labels = [f'Scar {scar_id}' for scar_id in range(17)]
        
        handles = []
        for task_id in np.unique(cfg.task_ids):
            task_id = int(task_id)
            plt.plot(range(len(cfg.task_ids)), task_performances[f"task_{task_id}"][f'{metric_name}_mean'], label=f"task_{task_id}", color=cfg.colors[task_id])
            plt.scatter(range(len(cfg.task_ids)), task_performances[f"task_{task_id}"][f'{metric_name}_mean'], marker=markers[task_id], c=cfg.colors[task_id])
            handles.append(mlines.Line2D([], [], marker=markers[task_id], linestyle='None', markersize=10, color=cfg.colors[task_id], label=dynamics_labels[task_id]))

        plt.legend(
            handles=handles,
            loc="lower center",
            ncol=1,
            bbox_to_anchor=(1.13, 0.125),
            fontsize=11
        )

        # Set horizontal gridlines
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

        # Set labels
        plt.xticks(ticks=range(len(cfg['task_ids'])), labels=range(len(cfg['task_ids'])), weight='bold')
        ax.set_ylabel(f"{metric_name.upper()}", labelpad=10, weight='bold', fontsize=12)
        ax.set_xlabel('Task #', labelpad=10, weight='bold', fontsize=12)
        ax.set_title(f"Model {metric_name.upper()} Performance Over Tasks", weight='bold', fontsize=15)

        plt.tight_layout()
        plt.savefig(f"{logger.log_dir}/temporal_{metric_name}{mode}_performance.png")
        plt.close()

    # Plot each metric in the config
    for metric in cfg.test_metrics:
        plot_metric(metric)

    # Save task performances to a text file
    json.dump(task_performances, fp=open(f"{logger.log_dir}/temporal_metrics.json", 'w'), indent=4)


def plot_tsne(cfg, logger, mode=""):
    # Load in embeddings
    embeddings, labels = [], []
    for idx, scar_id in enumerate(cfg.task_ids):
        embedding = np.load(f"{logger.log_dir}/task_{idx}/test_{idx}{mode}/test_{idx}{mode}_embeddings.npy", allow_pickle=True)        
        print(embedding.shape)
        
        embeddings.append(embedding)
        labels.append(np.full([embedding.shape[0]], fill_value=scar_id))

    # Pad embeddings to same length
    max_embed_length = 0
    for idx in range(len(embeddings)):
        if embeddings[idx].shape[1] > max_embed_length:
            max_embed_length = embeddings[idx].shape[1]
    
    for idx in range(len(embeddings)):
        if len(embeddings[idx].shape) == 3:
            embeddings[idx] = np.pad(embeddings[idx], pad_width=((0, 0), (0, max_embed_length - embeddings[idx].shape[1]), (0, 0)))
        else:
            embeddings[idx] = np.pad(embeddings[idx], pad_width=((0, 0), (0, max_embed_length - embeddings[idx].shape[1])))
    
    # Stack together
    embeddings = np.vstack(embeddings)
    embeddings = np.reshape(embeddings, [embeddings.shape[0], -1])
    labels = np.concatenate(labels)
    print(embeddings.shape, labels.shape)
    
    # Get tSNE embeddings
    tsne = TSNE(n_components=2, perplexity=50, metric="cosine", random_state=3, verbose=True)
    tsne_embedding = tsne.fit_transform(embeddings)

    # Plot figure
    plt.figure(figsize=(10, 5))

    # Plot the performances over tasks over time
    markers = ['o', 'v', '^', '<', '>', 's', '8', 'p', 'o', 'v', '^', '<', '>', 's', '8', 'p', 's', '8', 'p']
    dynamics_labels = [f'Scar {scar_id}' for scar_id in range(17)]
    
    handles = []
    for task_id in np.unique(cfg.task_ids):
        task_id = int(task_id)
        tsne_subset = tsne_embedding[np.where(labels == task_id)[0]]
        plt.scatter(tsne_subset[:, 0], tsne_subset[:, 1], marker=markers[task_id], c=cfg.colors[task_id])
        handles.append(mlines.Line2D([], [], marker=markers[task_id], linestyle='None', markersize=10, color=cfg.colors[task_id], label=dynamics_labels[task_id]))

    plt.legend(handles=handles, loc="lower center", ncol=1, bbox_to_anchor=(1.13, 0.125), fontsize=11)

    # Set labels
    plt.title(f"tSNE Meta-Embeddings over Scars", weight='bold', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{logger.log_dir}/meta_embedding_tsne{mode}.png", dpi=300)
    plt.close()
    

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

    # Shuffle task ids
    dynamic_tasks = cfg.task_ids
    np.random.shuffle(dynamic_tasks)

    # Make a consistent color theme across dynamic groups
    cfg.colors = []
    current_color = None
    for task_id in range(17):
        if task_id % 3 == 0:
            current_color = next(plt.gca()._get_lines.prop_cycler)['color']

        cfg.colors.append(current_color)

    print(f"=> Sequence of Tasks: {cfg.task_ids}")
    print(f"=> Sequence of Colors: {cfg.colors}")

    # Build datasets based on tasks
    datamodules = dict()
    for task_id in cfg.task_ids:
        datamodules[task_id] = ContinualEPDataModule(cfg, [task_id])
        print(f"=> Task {task_id}")
        print(f"=> Dataset 'train' xs shape: {datamodules[task_id].train_dataloader().dataset.xs.shape}")
        print(f"=> Dataset 'test' xs shape: {datamodules[task_id].test_dataloader().dataset.xs.shape}")

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

        # Training the model
        if cfg.train is True:
            trainer.fit(model, datamodules[task_id])
            if model.memory is not None:
                model.memory.update_logger(task_logger)
                model.memory.save_reservoir()
                
        # If we're just testing but its a MAML model, we have to initialize first
        elif cfg.train is not True and cfg.generate is True and cfg.model == "maml":
            trainer.fit_loop.max_steps = 1
            trainer.fit(model, datamodules[task_id])

        # Test on the training set
        if cfg.generate is True:
            cfg.split = "train"
            cfg.task_id = task_id
            trainer.test(model, datamodules[task_id].evaluate_train_dataloader(), ckpt_path=f"{task_logger.log_dir}/checkpoints/last.ckpt")

            # Test on all tasks
            for prev_task_idx, prev_task_id in enumerate(cfg.task_ids):
                cfg.split = f"{prev_task_idx}_train"
                trainer.test(model, datamodules[prev_task_id].evaluate_train_dataloader(), ckpt_path=f"{task_logger.log_dir}/checkpoints/last.ckpt")
                
                cfg.split = f"{prev_task_idx}"
                trainer.test(model, datamodules[prev_task_id], ckpt_path=f"{task_logger.log_dir}/checkpoints/last.ckpt")

        # If task boundaries are known, then reset the model's optimization state here
        if cfg.known_boundary is True and cfg.train is True:
            print("=> Known boundary, resetting optimizer state...")
            model.reset_state()

        # Plot continual metrics at this iteration
        plot_continual_metrics(cfg, logger, mode="_train")

        # Plot continual metrics at this iteration
        plot_continual_metrics(cfg, logger)

        # Remove preds and image npy files
        # os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_signals.npy' -delete")
        # os.system(f"find experiments/ -path 'experiments/{cfg.exptype}*' -name '*_preds.npy' -delete")
        
    # # Consolidate final metrics
    consolidate_metrics(cfg, logger, mode="")
    consolidate_metrics(cfg, logger, mode="_train")
    
    # Get tSNE over embeddings
    plot_tsne(cfg, logger, mode="")
    plot_tsne(cfg, logger, mode="_train")

if __name__ == '__main__':
    main()
