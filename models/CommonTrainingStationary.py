"""
@file CommonMetaDynamics.py

A common class that each meta latent dynamics function inherits.
Holds the training + validation step logic and the VAE components for reconstructions.
Has a testing step for holdout steps that handles all metric calculations and visualizations.
"""
import os
import json
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning

from utils import metrics
from utils.utils import get_memory
from models.CommonComponents import SpatialDecoder, get_params, Transition_Recurrent


class LatentMetaDynamicsModel(pytorch_lightning.LightningModule):
    def __init__(self, args):
        """ Generic training and testing boilerplate for the dynamics models """
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # Transition function
        self.propagation = Transition_Recurrent(args)
        
        # Decoder
        self.decoder = SpatialDecoder(args.num_filters, args.latent_dim)    

        # Losses
        self.reconstruction_loss = nn.MSELoss(reduction='none')

        # General trackers
        self.n_updates = 0

        # Accumulation of outputs over the logging interval
        self.outputs = list()

    def construct_nodes(self, data_idx, heart_name, data_path, batch_size, device, load_torso, load_physics, graph_method):
        """ Handles setting up the encoder/decoder node sizes for each heart mesh """
        params = get_params(data_path, heart_name, device, batch_size, load_torso, load_physics, graph_method)        
        self.domain.setup_nodes(data_idx, params)
        self.encoder.setup_nodes(data_idx, params)
        self.decoder.setup_nodes(data_idx, params)

    def forward(self, x, D, labels, generation_len):
        """ Placeholder function for the dynamics forward pass """
        raise NotImplementedError("In forward: Latent Dynamics function not specified.")

    def model_specific_loss(self, x, domain, train=True):
        """ Placeholder function for any additional loss terms a dynamics function may have """
        return 0.0

    def model_specific_plotting(self, version_path, outputs):
        """ Placeholder function for any additional plots a dynamics function may have """
        return None

    def configure_optimizers(self):
        """ Optimizer and LR scheduler """
        # Define optimizer
        optim = torch.optim.AdamW(list(self.parameters()), lr=self.args.learning_rate)

        # Define step optimizer
        optim_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.75)
        scheduler = {
            'scheduler': optim_scheduler,
            'interval': 'step'
        }
        return [optim], [scheduler]

    def on_train_start(self):
        """ Boilerplate experiment logging setup pre-training """
        # Get total number of parameters for the model and save
        self.log("total_num_parameters", float(sum(p.numel() for p in self.parameters() if p.requires_grad)), prog_bar=False)

        # Make image dir in lightning experiment folder if it doesn't exist
        if not os.path.exists(f"{self.logger.log_dir}/signals/"):
            os.mkdir(f"{self.logger.log_dir}/signals/")

    def get_step_outputs(self, batch, train=True):
        """ Handles processing a batch and getting model predictions """
        # Get batch
        x, x_domain, y, y_domain, names, labels = batch 
        
        # Changeable k_shot
        if train is True and self.args.domain_varying is True:
            k_shot = np.random.randint(1, x_domain.shape[1])
            x_domain = x_domain[:, :k_shot]
            y_domain = y_domain[:, :k_shot]

        print(f"X pre: {x.shape} | Domain pre: {x_domain.shape}| Name pre: {names.shape} {names[0]}")

        # Get predictions
        preds, zt = self(x, x_domain, y, y_domain, int(names[0]))

        # Get the likelihood
        nll_raw = self.reconstruction_loss(preds, x)
        nll_0 = nll_raw[:, :, 0].sum()
        nll_r = nll_raw[:, :, 1:].sum() / (x.shape[-1] - 1)
        likelihood = x.shape[-1] * (nll_0 * 0.1 + nll_r)
            
        # Get the loss terms from the specific latent dynamics loss
        model_specific_loss = self.model_specific_loss(x, x_domain, preds)
        return x, preds, zt, names, likelihood, model_specific_loss

    def get_metrics(self, outputs, setting):
        """ Takes the dictionary of saved batch metrics, stacks them, and gets outputs to log in the Tensorboard """
        # Pad to longest vertice length
        print(f'Pre pad: {[out["signals"].shape[1] for out in outputs]}')
        max_x_vertice = max([out["signals"].shape[1] for out in outputs])
        for idx, out in enumerate(outputs):
            if max_x_vertice - outputs[idx]["signals"].shape[1] > 0:
                outputs[idx]["signals"] = torch.nn.functional.pad(outputs[idx]["signals"], pad=[0, 0, 0, max_x_vertice - outputs[idx]["signals"].shape[1], 0, 0], mode='constant', value=0)       
                outputs[idx]["preds"] = torch.nn.functional.pad(outputs[idx]["signals"], pad=[0, 0, 0, max_x_vertice - outputs[idx]["signals"].shape[1], 0, 0], mode='constant', value=0)       
        print(f'Post pad: {[out["signals"].shape[1] for out in outputs]}')
        
        # Convert outputs to Tensors and then Numpy arrays
        signals = torch.vstack([out["signals"] for out in outputs]).cpu().numpy()
        preds = torch.vstack([out["preds"] for out in outputs]).cpu().numpy()

        # Iterate through each metric function and add to a dictionary
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            out_metrics[met] = metric_function(signals, preds, args=self.args, setting=setting)[1]
        return out_metrics

    def training_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """        
        # Get outputs and calculate losses
        signals, preds, _, names, likelihood, model_specific_loss  = self.get_step_outputs(batch)

        # Modulate total loss
        loss = likelihood + model_specific_loss

        # Build the full loss
        self.log_dict({"likelihood": likelihood}, prog_bar=True)
        
        # Sample new task for next batch
        self.trainer.train_dataloader.dataset.datasets.sample_new_task()
        
        # Return outputs as dict
        self.n_updates += 1
        self.outputs.append({"preds": preds.detach().cpu(), "signals": signals.detach().cpu(), "names": names.detach().cpu()})
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """ Every N  Steps, perform the SOM optimization """
        # Log metrics over saved batches on the specified interval
        if batch_idx % self.args.log_interval == 0 and batch_idx != 0:
            metrics = self.get_metrics(self.outputs, setting='train')
            for metric in metrics.keys():
                self.log(f"train_{metric}", metrics[metric], prog_bar=True)

            # Wipe the saved batch outputs
            self.outputs = list()
            
    def validation_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """
        # Get outputs and calculate losses
        signals, preds, _, _, likelihood, _  = self.get_step_outputs(batch, train=False)

        # Build the full loss
        self.log_dict({"val_likelihood": likelihood}, prog_bar=True)

        # Return outputs as dict
        return {"loss": likelihood, "signals": signals.detach().cpu(), "preds": preds.detach().cpu()}
    
    def validation_epoch_end(self, outputs):
        """
        Every N epochs, get a validation reconstruction sample
        :param outputs: list of outputs from the validation steps at batch 0
        """
        # Log epoch metrics on saved batches
        metrics = self.get_metrics(self.outputs, setting='val')
        for metric in metrics.keys():
            self.log(f"val_{metric}", metrics[metric], prog_bar=True)

    def test_step(self, batch, batch_idx):
        """ PyTorch-Lightning testing step """
        # Get model outputs from batch
        signals, preds, _, _, _, _ = self.get_step_outputs(batch, train=False)
        print(signals.shape, preds.shape)

        # Return output dictionary
        out = dict()
        for key, item in zip(["preds", "signals"], [preds, signals]):
            out[key] = item.detach().cpu().numpy()

        # Add meta-embeddings if it is a meta-model
        if self.args.meta is True:
            out["embeddings"] = self.embeddings.detach().cpu().numpy()

        return out

    def test_epoch_end(self, batch_outputs):
        """ For testing end, save the predictions, gt, and MSE to NPY files in the respective experiment folder """
        # Set up output path and create dir
        output_path = f"{self.logger.log_dir}/test_{self.args.split}"
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Stack all output types and convert to numpy
        outputs = dict()
        for key in batch_outputs[0].keys():
            outputs[key] = np.vstack([output[key] for output in batch_outputs])

        # Save to files
        if self.args.save_files is True:
            for key in outputs.keys():
                np.save(f"{output_path}/test_{self.args.split}_{key}.npy", outputs[key])

        # Iterate through each metric function and add to a dictionary
        print("\n=> getting metrics...")
        out_metrics = {}
        for met in self.args.metrics:
            metric_function = getattr(metrics, met)
            metric_results, metric_mean, metric_std = metric_function(outputs["signals"], outputs["preds"], args=self.args, setting='test')
            out_metrics[f"{met}_mean"], out_metrics[f"{met}_std"] = float(metric_mean), float(metric_std)
            print(f"=> {met}: {metric_mean:4.5f}+-{metric_std:4.5f}")

        # Save metrics to JSON in checkpoint folder
        with open(f"{output_path}/test_{self.args.split}_metrics.json", 'w') as f:
            json.dump(out_metrics, f)

        # Save metrics to an easy excel conversion style
        with open(f"{output_path}/test_{self.args.dataset}_excel.txt", 'w') as f:
            for metric in self.args.metrics:
                f.write(f"{out_metrics[f'{metric}_mean']:0.3f}({out_metrics[f'{metric}_std']:0.3f}),")
