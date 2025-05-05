"""
@file dataloader.py

Holds the Datasets and Dataloaders
"""
import torch
import numpy as np
import pytorch_lightning

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, RandomSampler


class ContinualEPDataset(Dataset):
    """ Basic Dataset object for the SSM """

    def __init__(self, args, xs, ys, labels, scars, names):
        self.args = args
        self.xs = xs
        self.ys = ys
        self.labels = labels
        self.scars = scars
        self.names = names
        
    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        # Get indices of context and query
        # TODO: change to .choice
        indices = np.random.randint(0, self.xs.shape[0], 1 + self.args.domain_size)
        query_idx = indices[0]
        context_idx = indices[1:]
        
        # Get the query
        xs_qry = self.xs[query_idx, :]
        # ys_qry = self.ys[query_idx, :]
        ys_qry = self.labels[:, query_idx]
        
        label = self.labels[:, query_idx]
        name = self.names[query_idx]
        scar = self.scars[query_idx]

        # Get the corresponding context set
        xs_spt = self.xs[context_idx, :]
        # ys_spt = self.ys[context_idx, :]
        ys_spt = self.labels[:, context_idx].permute(1, 0).contiguous()
        return xs_qry, xs_spt, ys_qry, ys_spt, name, label, scar


class ContinualEPDataModule(pytorch_lightning.LightningDataModule):
    """ Custom DataModule object that handles preprocessing all sets of data for a given run """
    def __init__(self, args, task_ids):
        super(ContinualEPDataModule, self).__init__()
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
            
        if self.args.omit is not None:
            xs = xs[:, :, self.args.omit:]
        
        labels = npzfile['label'].astype(int)
        scar = npzfile['scar']
        names = npzfile['heart_name']

        # Convert to Tensors
        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()
        labels = torch.from_numpy(labels)
        names = torch.from_numpy(names)
        scar = torch.from_numpy(scar)

        # Build dataset and corresponding Dataloader
        dataset = ContinualEPDataset(self.args, xs, ys, labels, scar, names)

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


class StationaryEPDataset(Dataset):
    """ Basic Dataset object for the SSM """
    def __init__(self, args, task_ids, xs, labels, scars, names):
        self.args = args
        self.task_ids = task_ids
        
        # First set up macro dictionary keys based on heart mesh [AW, DC, EC]
        self.datasets = {}
        for heart_name in range(len(args.data_names)):
            self.datasets[heart_name] = {}
            
        # Then for each scar populate its relevant heart mesh
        for idx, scar in enumerate(task_ids):
            self.datasets[int(names[idx][0])][scar] = {
                'xs': xs[idx],
                'label': labels[idx],
                'scar': scars[idx],
                'name': names[idx]
            }
        
        # Set first episode to AW
        self.current_task = 0
        
    def __len__(self):
        return len(self.args.data_names)

    def __getitem__(self, idx):
        # Get relevant heart mesh from idx
        heart_mesh_dict = self.datasets[self.current_task]
        
        # Sample scars and relevant dataset
        scar_idx = np.random.choice(list(heart_mesh_dict.keys()))
        xs = heart_mesh_dict[scar_idx]['xs']
        labels = heart_mesh_dict[scar_idx]['label']
        names = heart_mesh_dict[scar_idx]['name']

        # Get indices of context and query
        indices = np.random.randint(0, xs.shape[0], 1 + self.args.domain_size)
        query_idx = indices[0]
        context_idx = indices[1:]
        
        # Get the query
        xs_qry = xs[query_idx, :]
        ys_qry = labels[:, query_idx]
        
        label = labels[:, query_idx]
        name = names[query_idx]

        # Get the corresponding context set
        xs_spt = xs[context_idx, :]
        ys_spt = labels[:, context_idx].permute(1, 0).contiguous()
        return xs_qry, xs_spt, ys_qry, ys_spt, name, label

    def sample_new_task(self):
        self.current_task = np.random.choice(range(len(self.args.data_names)))


class StationaryTestEPDataset(Dataset):
    """ Basic Dataset object for the SSM """
    def __init__(self, args, xs, ys, labels, scars, names):
        self.args = args
        self.xs = xs
        self.ys = ys
        self.labels = labels
        self.scars = scars
        self.names = names
        
    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        # Get indices of test set
        indices = [i for i in range(self.xs.shape[0])]
        indices.remove(idx)
        
        # Build context and queries indices based on which one was chosen
        query_idx = idx
        context_idx = np.random.choice(indices, min(self.args.domain_size, len(indices)), replace=False)
        print(query_idx, context_idx)
        
        # # Get indices of context and query
        # indices = np.random.randint(0, self.xs.shape[0], 1 + self.args.domain_size)
        # query_idx = idx
        # context_idx = indices[1:]
        
        # Get the query
        xs_qry = self.xs[query_idx, :]
        # ys_qry = self.ys[query_idx, :]
        ys_qry = self.labels[:, query_idx]
        
        label = self.labels[:, query_idx]
        name = self.names[query_idx]

        # Get the corresponding context set
        xs_spt = self.xs[context_idx, :]
        ys_spt = self.labels[:, context_idx].permute(1, 0).contiguous()
        return xs_qry, xs_spt, ys_qry, ys_spt, name, label


class StationaryEPDataModule(pytorch_lightning.LightningDataModule):
    """ Custom DataModule object that handles preprocessing all sets of data for a given run """
    def __init__(self, args, task_ids):
        super(StationaryEPDataModule, self).__init__()
        self.args = args
        self.task_ids = task_ids

    def make_loader(self, mode="train", evaluation=True, shuffle=True):
        xs_list, label_list, scar_list, name_list = [], [], [], []
        
        # Iterate over task ids and stack datasets
        for scar in self.task_ids:    
            npzfile = np.load(f"data/{self.args.dataset}/{self.args.dataset_ver}{scar}/{mode}.npz")

            # Load in data sources
            xs = npzfile['xs']
            xs = np.swapaxes(xs, 2, 1)
            
            if self.args.window is not None:
                xs = xs[:, :, :self.args.window]
                
            if self.args.omit is not None:
                xs = xs[:, :, self.args.omit:]
            
            labels = npzfile['label'].astype(int)
            scar = npzfile['scar']
            names = npzfile['heart_name']

            # Convert to Tensors
            xs = torch.from_numpy(xs).float()
            labels = torch.from_numpy(labels)
            names = torch.from_numpy(names)
            scar = torch.from_numpy(scar)
            print(xs.shape, labels.shape)
            
            xs_list.append(xs)
            label_list.append(labels)
            name_list.append(names)
            scar_list.append(scar)

        # Build dataset and corresponding Dataloader
        dataset = StationaryEPDataset(self.args, self.task_ids, xs_list, label_list, scar_list, name_list)

        # Build dataloader based on whether it is training or evaluation
        if evaluation is False:
            sampler = RandomSampler(data_source=dataset, replacement=True, num_samples=self.args.num_steps * self.args.batch_size)
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=self.args.num_workers, pin_memory=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, num_workers=self.args.num_workers, pin_memory=True)
        return dataloader

    def make_test_loader(self, task_id, mode="train"):
        # Iterate over task ids and stack datasets
        npzfile = np.load(f"data/{self.args.dataset}/{self.args.dataset_ver}{task_id}/{mode}.npz")

        # Load in data sources
        xs = npzfile['xs']
        ys = npzfile['ys']
        xs = np.swapaxes(xs, 2, 1)
        
        if self.args.window is not None:
            xs = xs[:, :, :self.args.window]
            
        if self.args.omit is not None:
            xs = xs[:, :, self.args.omit:]
        
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
        dataset = StationaryTestEPDataset(self.args, xs, ys, labels, scar, names)

        # Build dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        return dataloader

    def train_dataloader(self):
        """ Getter function that builds and returns the training dataloader """
        return self.make_loader("train", evaluation=False)

    def evaluate_train_dataloader(self):
        return self.make_loader("train", shuffle=False)

    # def val_dataloader(self):
    #     """ Getter function that builds and returns the validation dataloader """
    #     return self.make_loader("val", shuffle=False)

    def test_dataloader(self, task_id, mode="val"):
        """ Getter function that builds and returns the testing dataloader """
        return self.make_test_loader(task_id, mode)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    from utils import flatten_cfg
    
    @hydra.main(version_base="1.3", config_path="/home/rxm7244/Projects/CoSFan-NeuralSurrogate/configs", config_name="stationary")
    def main(cfg: DictConfig):
        cfg.exptype = cfg.exptype
        cfg = flatten_cfg(cfg)
        
        dataloader = StationaryEPDataModule(cfg, task_ids=cfg.task_ids).train_dataloader()
        
        idx = 0
        for sample in dataloader:
            print(sample[0].shape, sample[1].shape, dataloader.dataset.current_task)
            dataloader.dataset.sample_new_task()

            idx += 1
            
            if idx > 10:
                break
    
    main()