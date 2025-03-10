import torch
import numpy as np

from memory._Memory import Memory


class ExactReplay(Memory):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # Dictionary of previous tasks and exemplars
        self.tasks = dict()

    def task_update(self):
        pass

    def batch_update(self, xs, names, task_counter=None):
        pass

    def epoch_update(self, _, task_id, model=None):
        # Get full dataset for this task
        xs = model.trainer.train_dataloader.dataset.datasets.xs.to(self.args.devices[0])
        names = model.trainer.train_dataloader.dataset.datasets.names.to(self.args.devices[0])
        labels = model.trainer.train_dataloader.dataset.datasets.labels.permute(1, 0).contiguous().to(self.args.devices[0])
        scars = model.trainer.train_dataloader.dataset.datasets.scars.to(self.args.devices[0])

        if self.args.window is not None:
            xs = xs[:, :, :self.args.window]

        # Assign to the dictionary
        self.tasks[task_id] = {
            'xs': xs,
            'names': names,
            'labels': labels,
            'scars': scars
        }
        print(f"Task Distribution: {self.tasks.keys()} {[self.tasks[ti]['xs'].shape for ti in self.tasks.keys()]}")

    def get_batch(self):
        # Get proportion of tasks by batch size
        task_size = max((self.args.batch_size // 2) // (len(list(self.tasks.keys()))), 2)
        # print(f"Task size: {task_size} | Task Keys: {self.tasks.keys()}")

        # Build outputs of each task
        xs, domains, names = [], [], []
        ys, ys_domains = [], []
        for task_id in self.tasks.keys():
            sample_indices = np.random.choice(range(self.tasks[task_id]['xs'].shape[0]), task_size + self.args.domain_size, replace=False)

            xs.extend([s for s in self.tasks[task_id]['xs'][sample_indices][self.args.domain_size:]])
            names.append(self.tasks[task_id]['names'][sample_indices][self.args.domain_size:])
            domains.extend([s for s in self.tasks[task_id]['xs'][sample_indices][:self.args.domain_size].unsqueeze(0).repeat(task_size, 1, 1, 1)])
            
            ys.extend([s for s in self.tasks[task_id]['labels'][sample_indices][self.args.domain_size:]])
            ys_domains.extend([s for s in self.tasks[task_id]['labels'][sample_indices][:self.args.domain_size].unsqueeze(0).repeat(task_size, 1, 1)])
  
        names = torch.vstack(names)
        # print(f"Names from ER: {names}")
        return xs, domains, ys, ys_domains, names
    
    # def get_batch(self):
    #     task_size = self.args.batch_size // 2
        
    #     # Build outputs of each task
    #     xs, xs_domains, names = [], [], []
    #     ys, ys_domains = [], []
    #     task_sequence = np.random.choice(len(self.tasks.keys()), size=task_size, replace=True)
    #     for task_id in task_sequence:
    #         index = np.random.choice(range(self.tasks[task_id]['xs'].shape[0]), 1 + self.args.domain_size, replace=False)
            
    #         xs.append(self.tasks[task_id]['xs'][index][0])
    #         ys.append(self.tasks[task_id]['labels'][index][0])
    #         names.append(self.tasks[task_id]['names'][index][0])
            
    #         xs_domains.append(self.tasks[task_id]['xs'][index][1:])
    #         ys_domains.append(self.tasks[task_id]['labels'][index][1:])
        
    #     names = torch.vstack(names)
    #     return xs, xs_domains, ys, ys_domains, names
