import torch
import numpy as np

from memory._Memory import Memory


class TaskAwareReservoir(Memory):
    def __init__(self, args):
        super().__init__(args)

    def epoch_update(self, logger, task_counter, model=None):
        print(self.current_task_indices, len(self.buffer['xs']), self.buffer['ys'].shape)
        
        print([x.shape for x in self.buffer['xs']])
        # Assign new samples to reservoir
        for idx, base_idx in enumerate(self.current_task_indices):
            self.buffer['xs'][base_idx] = self.current_task_data['xs'][idx]
        
        self.buffer["ys"][self.current_task_indices] = self.current_task_data['ys']
        self.buffer["names"][self.current_task_indices] = self.current_task_data['names']
        self.buffer["scars"][self.current_task_indices] = self.current_task_data['scars']
        
        # Wipe the accumulated last-task
        self.current_task_indices = None
        self.current_task_data['xs'] = None
        self.current_task_data['ys'] = None
        self.current_task_data['names'] = None
        self.current_task_data['scars'] = None

        # Print out the current distribution to stdout
        print(f"=> Distribution of scars: {np.unique(self.buffer['scars'].detach().cpu().numpy(), return_counts=True)}")

    def batch_update(self, xs, ys, names, scars, task_counter=None):
         # Initialize vectors at first batch
        if self.age == 0:
            self.buffer["xs"] = [xs[0]]
            self.buffer["ys"] = ys[0].unsqueeze(0)
            self.buffer["names"] = names[0].unsqueeze(0)
            self.buffer["scars"] = scars[0].unsqueeze(0)
            self.age += 1

        # Just add to buffer if buffer is not filled
        elif self.args.memory_samples - self.age > 0:
            stopping_point = self.args.memory_samples - self.age
            
            for sample in xs[:stopping_point]:
                self.buffer['xs'].append(sample)
            
            # self.buffer["xs"] = torch.vstack((self.buffer["xs"], xs[:stopping_point]))
            self.buffer["ys"] = torch.vstack((self.buffer["ys"], ys[:stopping_point]))
            self.buffer["names"] = torch.vstack((self.buffer["names"], names[:stopping_point]))
            self.buffer["scars"] = torch.concatenate((self.buffer["scars"], scars[:stopping_point]))
            self.age += self.args.batch_size // 2

        # Otherwise take a proportional sample and add if it is under threshold
        else:
            p = torch.randint(0, self.age, [self.args.batch_size // 2])
            indices = torch.where(p < self.args.memory_samples)[0]
            p = p[indices]

            # If current task buffer is not initialized, just assign values
            if self.current_task_indices is None:
                self.current_task_indices = p
                self.current_task_data["xs"] = xs[indices]
                self.current_task_data["ys"] = ys[indices]
                self.current_task_data["names"] = names[indices]
                self.current_task_data["scars"] = scars[indices]

            # Other append *only* unique indices thus far, skip repeats
            else:
                # Find the non-intersecting set between the already set indices and new ones
                non_insect = p[(p[:, None] != self.current_task_indices).all(dim=1)]

                # Then get those indices from the indice set
                indices_to_use = []
                for non_i in non_insect:
                    indices_to_use.append(np.where(p == non_i)[0][0])
                indices_to_use = np.array(indices_to_use)

                # Add to current task buffer
                # print(xs[indices_to_use].shape, ys[indices_to_use].shape, names[indices_to_use].shape, scars[indices_to_use].shape)
                self.current_task_indices = torch.concatenate((self.current_task_indices, p[indices_to_use]))
                self.current_task_data["xs"] = torch.vstack((self.current_task_data["xs"], xs[indices_to_use]))
                self.current_task_data["ys"] = torch.vstack((self.current_task_data["ys"], ys[indices_to_use]))
                self.current_task_data["names"] = torch.vstack((self.current_task_data["names"], names[indices_to_use]))
                self.current_task_data["scars"] = torch.concatenate((self.current_task_data["scars"], scars[indices_to_use]))

            self.age += self.args.batch_size // 2

    def get_batch(self, k_shot):
        # Select random indices from the reservoir
        sample_indices = np.random.choice(range(len(self.buffer["xs"])), self.args.batch_size // 2, replace=False)

        scars_base = self.buffer['scars'].cpu()

        # Get the corresponding data
        xs, xs_domains, names, scars = [], [], [], []
        ys, ys_domains = [], []
        for sample_idx in sample_indices:
            # Get query
            xs.append(self.buffer['xs'][sample_idx])
            ys.append(self.buffer['ys'][sample_idx])
            names.append(self.buffer['names'][sample_idx])
            scars.append(self.buffer['scars'][sample_idx])
            
            # Get context
            domain_indices = np.where(scars_base == scars_base[sample_idx])[0]
            domain_indices = domain_indices[np.where(domain_indices != sample_idx)[0]]
            domain_sample_indices = np.random.choice(domain_indices, k_shot, replace=True)
            
            xs_domains.append(torch.stack([self.buffer['xs'][idx] for idx in domain_sample_indices]))
            ys_domains.append(torch.stack([self.buffer['ys'][idx] for idx in domain_sample_indices]))
        
        # Stack the domains
        names = torch.vstack(names)
        scars = torch.vstack(scars)
        # print(torch.unique(scars), )
        return xs, xs_domains, ys, ys_domains, names, scars
