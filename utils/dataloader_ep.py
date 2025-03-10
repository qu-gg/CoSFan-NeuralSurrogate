"""
@file dataloader.py

Holds the Datasets and Dataloaders
"""
import torch
import numpy as np
import pytorch_lightning

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler


class EPDataset(Dataset):
    """ Basic Dataset object for the SSM """

    def __init__(self, args, xs, ys, labels, scars, names):
        self.args = args
        self.xs = xs
        self.ys = ys
        self.labels = labels
        self.scars = scars
        self.names = names
        
    def __len__(self):
        return self.xs.shape[0] - self.args.domain_size

    def __getitem__(self, idx):
        # Get indices of context and query
        indices = np.random.randint(0, self.xs.shape[0], 1 + self.args.domain_size)
        query_idx = indices[0]
        context_idx = indices[1:]
        
        # Get the query
        xs_qry = self.xs[query_idx, :]
        # ys_qry = self.ys[query_idx, :]
        ys_qry = self.labels[:, query_idx]
        
        label = self.labels[:, query_idx]
        name = self.names[query_idx]

        # Get the corresponding context set
        xs_spt = self.xs[context_idx, :]
        # ys_spt = self.ys[context_idx, :]
        ys_spt = self.labels[:, context_idx].permute(1, 0).contiguous()
        return xs_qry, xs_spt, ys_qry, ys_spt, name, label


class EPDataModule(pytorch_lightning.LightningDataModule):
    """ Custom DataModule object that handles preprocessing all sets of data for a given run """
    def __init__(self, args, task_ids):
        super(EPDataModule, self).__init__()
        self.args = args
        self.task_ids = task_ids

    def make_loader(self, mode="train", evaluation=True, shuffle=True):
        # Iterate over task ids and stack datasets
        npzfile = np.load(f"data/{self.args.dataset}/{self.args.dataset_ver}{self.task_ids[0]}/{mode}.npz")

        # Load in data sources
        xs = npzfile['xs']
        ys = npzfile['ys']
        xs = np.swapaxes(xs, 2, 1)
        
        if self.args.window is not None:
            xs = xs[:, :, :self.args.window]
        
        labels = npzfile['label'].astype(int)
        scar = npzfile['scar']
        names = npzfile['heart_name']

        # Convert to Tensors
        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()
        labels = torch.from_numpy(labels)
        names = torch.from_numpy(names)
        scar = torch.from_numpy(scar)
        print(xs.shape, labels.shape)

        # Build dataset and corresponding Dataloader
        dataset = EPDataset(self.args, xs, ys, labels, scar, names)

        # Build dataloader based on whether it is training or evaluation
        if evaluation is False:
            sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=self.args.num_task_steps * self.args.batch_size)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=self.args.num_workers, pin_memory=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=True)
        return dataloader

    def train_dataloader(self):
        """ Getter function that builds and returns the training dataloader """
        return self.make_loader("train", evaluation=False)

    def evaluate_train_dataloader(self):
        return self.make_loader("train", shuffle=False)

    # def val_dataloader(self):
    #     """ Getter function that builds and returns the validation dataloader """
    #     return self.make_loader("val", shuffle=False)

    def test_dataloader(self):
        """ Getter function that builds and returns the testing dataloader """
        return self.make_loader("val", shuffle=False)
