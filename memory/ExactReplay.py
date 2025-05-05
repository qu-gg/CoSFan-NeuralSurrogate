import torch
import numpy as np

from memory._Memory import Memory


class ExactReplay(Memory):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # Dictionary of previous tasks and exemplars
        self.buffer = {
            'xs': [],
            'ys': None,
            'names': None,
            'scars': None
        }

    def task_update(self):
        pass

    def batch_update(self, xs, ys, names, scars, task_counter=None):
        pass

    def epoch_update(self, _, task_id, model=None):
        if len(self.buffer['xs']) == 0:
            self.buffer['xs'] = [x for x in model.trainer.train_dataloader.dataset.datasets.xs.to(self.args.devices[0])]
            self.buffer['ys'] = model.trainer.train_dataloader.dataset.datasets.labels.permute(1, 0).to(self.args.devices[0])
            self.buffer['names'] = model.trainer.train_dataloader.dataset.datasets.names.to(self.args.devices[0])
            self.buffer['labels'] = model.trainer.train_dataloader.dataset.datasets.labels.permute(1, 0).contiguous().to(self.args.devices[0])
            self.buffer['scars'] = model.trainer.train_dataloader.dataset.datasets.scars.to(self.args.devices[0])
        else:
            self.buffer['xs'].extend([x for x in model.trainer.train_dataloader.dataset.datasets.xs.to(self.args.devices[0])])
            self.buffer['ys'] = torch.vstack((self.buffer['ys'], model.trainer.train_dataloader.dataset.datasets.labels.permute(1, 0).contiguous().to(self.args.devices[0])))
            self.buffer['names'] = torch.vstack((self.buffer['names'], model.trainer.train_dataloader.dataset.datasets.names.to(self.args.devices[0])))
            self.buffer['labels'] = torch.vstack((self.buffer['labels'], model.trainer.train_dataloader.dataset.datasets.labels.permute(1, 0).contiguous().to(self.args.devices[0])))
            self.buffer['scars'] = torch.concatenate((self.buffer['scars'], model.trainer.train_dataloader.dataset.datasets.scars.to(self.args.devices[0])))
            
        print(len(self.buffer['xs']))
        print(self.buffer['ys'].shape)
        print(self.buffer['names'].shape)
        print(self.buffer['labels'].shape)
        print(self.buffer['scars'].shape)
            
        print(f"Task Distribution: {torch.unique(self.buffer['scars'], return_counts=True)}")

    # def get_batch(self, k_shot):
    #     # Select random indices from the reservoir
    #     sample_indices = np.random.choice(range(len(self.buffer["xs"])), self.args.batch_size // 2, replace=False)

    #     scars_base = self.buffer['scars'].cpu()

    #     # Get the corresponding data
    #     xs, xs_domains, names, scars = [], [], [], []
    #     ys, ys_domains = [], []
    #     for sample_idx in sample_indices:
    #         # Get query
    #         xs.append(self.buffer['xs'][sample_idx])
    #         ys.append(self.buffer['ys'][sample_idx])
    #         names.append(self.buffer['names'][sample_idx])
    #         scars.append(self.buffer['scars'][sample_idx])
            
    #         # Get context
    #         domain_indices = np.where(scars_base == scars_base[sample_idx])[0]
    #         domain_indices = domain_indices[np.where(domain_indices != sample_idx)[0]]
    #         domain_sample_indices = np.random.choice(domain_indices, k_shot, replace=True)
            
    #         xs_domains.append(torch.stack([self.buffer['xs'][idx] for idx in domain_sample_indices]))
    #         ys_domains.append(torch.stack([self.buffer['ys'][idx] for idx in domain_sample_indices]))
        
    #     # Stack the domains
    #     names = torch.vstack(names)
    #     scars = torch.vstack(scars)
    #     # print(torch.unique(scars), )
    #     return xs, xs_domains, ys, ys_domains, names, scars
    
    def get_batch(self, k_shot):
        # Get unique task IDs and create task-specific indices
        scars_base = self.buffer['scars'].cpu()
        unique_tasks = torch.unique(scars_base)
        task_indices = {int(task): torch.where(scars_base == task)[0].numpy() for task in unique_tasks}
        
        # Sample tasks with replacement for batch_size // 2 queries
        sampled_tasks = np.random.choice(unique_tasks.numpy(), size=self.args.batch_size // 2, replace=True)
        
        # Initialize lists to store batch data
        xs, xs_domains, names, scars = [], [], [], []
        ys, ys_domains = [], []
        
        for task_id in sampled_tasks:
            # Get indices for this task
            task_specific_indices = task_indices[int(task_id)]
            
            # Sample one query example from this task
            query_idx = np.random.choice(task_specific_indices, size=1)[0]
            
            # Get query data
            xs.append(self.buffer['xs'][query_idx])
            ys.append(self.buffer['ys'][query_idx])
            names.append(self.buffer['names'][query_idx])
            scars.append(self.buffer['scars'][query_idx])
            
            # Sample k_shot context examples from the same task (excluding query)
            context_indices = task_specific_indices[task_specific_indices != query_idx]
            domain_sample_indices = np.random.choice(context_indices, k_shot, replace=True)
            
            # Get context data
            xs_domains.append(torch.stack([self.buffer['xs'][idx] for idx in domain_sample_indices]))
            ys_domains.append(torch.stack([self.buffer['ys'][idx] for idx in domain_sample_indices]))
        
        # Stack the domains
        names = torch.vstack(names)
        scars = torch.vstack(scars)
        
        return xs, xs_domains, ys, ys_domains, names, scars