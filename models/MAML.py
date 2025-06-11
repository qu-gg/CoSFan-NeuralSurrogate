from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np

from models.CommonComponents import get_params, one_hot_label, SpatialEncoder
from models.CommonTraining import LatentMetaDynamicsModel


class Transition_Recurrent(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.z_dim = args.latent_dim
        self.transition_dim = args.num_hidden
        self.identity_init = True
        self.stochastic = False

        # compute the gain (gate) of non-linearity
        self.lin1 = nn.Linear(self.z_dim, self.transition_dim*2)
        self.lin2 = nn.Linear(self.transition_dim*2, self.z_dim)
        # compute the proposed mean
        self.lin3 = nn.Linear(self.z_dim, self.transition_dim*2)
        self.lin4 = nn.Linear(self.transition_dim*2, self.z_dim)
        # compute the linearity part
        self.lin_m = nn.Linear(self.z_dim, self.z_dim)
        self.lin_n = nn.Linear(self.z_dim, self.z_dim)

        if self.identity_init:
            self.lin_n.weight.data = torch.eye(self.z_dim)
            self.lin_n.bias.data = torch.zeros(self.z_dim)

        # compute the logvar
        self.lin_v = nn.Linear(self.z_dim, self.z_dim)
        # logvar activation
        self.act_var = nn.Tanh()

        self.act_weight = nn.Sigmoid()
        self.act = nn.ELU()
    
    def init_z_0(self, trainable=True):
        return nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable), \
            nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable)
    
    def forward(self, t, z_t_1):
        z_combine = z_t_1
        _g_t = self.act(self.lin1(z_combine))
        g_t = self.act_weight(self.lin2(_g_t))
        _h_t = self.act(self.lin3(z_combine))
        h_t = self.act(self.lin4(_h_t))
        _mu = self.lin_m(z_combine)
        mu = (1 - g_t) * self.lin_n(_mu) + g_t * h_t
        mu = mu + _mu

        if self.stochastic:
            _var = self.lin_v(h_t)
            # if self.clip:
            #     _var = torch.clamp(_var, min=-100, max=85)
            var = self.act_var(_var)
            return mu, var
        else:
            return mu


class Maml(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)
        self.automatic_optimization = False

        # Transition function
        self.propagation = Transition_Recurrent(args)

        # encoder
        self.signal_encoder = SpatialEncoder(args.num_filters, args.latent_dim, cond=True)

        # initialization
        self.condition_encoder = SpatialEncoder(args.num_filters, args.latent_dim)
        self.initial = nn.Linear(args.latent_dim, args.latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def construct_nodes(self, data_idx, heart_name, data_path, batch_size, k_shot, device, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, device, batch_size, load_torso, load_physics, graph_method)        
        self.bg[data_idx] = params["bg"]
        self.bg1[data_idx] = params["bg1"]
        self.bg2[data_idx] = params["bg2"]
        self.bg3[data_idx] = params["bg3"]
        self.bg4[data_idx] = params["bg4"]
        
        self.signal_encoder.setup_nodes(data_idx, params)
        self.condition_encoder.setup_nodes(data_idx, params)
        self.decoder.setup_nodes(data_idx, params)

    def latent_initial(self, y, N, V, heart_name):
        y = one_hot_label(y[:, 2] - 1, N, V, 1, self.args.devices[0])
        y = y[:, :].view(N, V, 1)
        z_0 = self.condition_encoder(y, heart_name)
        z_0 = torch.squeeze(z_0)
        z_0 = self.initial(z_0)
        
        return z_0

    def time_modeling(self, T, z_0):
        # print(z_0.shape)
        N, V, C = z_0.shape


        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(T, z_prev)
            z_prev = z_t
            z_t = z_t.view(1, N, V, C)
            z.append(z_t)
        z = torch.cat(z, dim=0)
        z_0 = z_0.view(1, N, V, C)
        z = torch.cat([z_0, z], dim=0)
        
        # elif self.trans_model in ['ODE',]:
            # z = self.propagation(T, z_0, z_c)
        z = z.permute(1, 2, 3, 0).contiguous()
        return z

    def forward(self, x, xD, y, yD, name):
        N, V, T = x.shape

        # Initial state z0
        z0 = self.latent_initial(y, N, V, name)
        if z0.dim() == 2:
            z0 = z0.unsqueeze(0)
        
        # Latent propagation
        z = self.time_modeling(T, z0)
        
        # Decode
        x_ = self.decoder(z, name)   
        return x_, z

    def single_fast_weight(self, x, xD, y, yD, name):
        optim = torch.optim.SGD(list(self.parameters()), lr=self.args.learning_rate)
        
        """ Inner step """
        for _ in range(self.args.inner_steps):
            pred, _ = self(x, xD, y, yD, int(name))
            
            # Likelihood
            nll_raw = self.reconstruction_loss(pred, x)
            nll_0 = nll_raw[:, :, 0].sum()
            nll_r = nll_raw[:, :, 1:].sum() / (x.shape[-1] - 1)
            loss = x.shape[-1] * (nll_0 * 0.1 + nll_r)
            
            # Get the loss terms from the specific latent dynamics loss
            model_specific_loss = self.model_specific_loss(x, xD, pred)
            likelihood = loss + model_specific_loss                

            # Get loss and gradients
            optim.zero_grad(set_to_none=True)
            likelihood.backward()
            optim.step()

            # Perform SGD over the dynamics function parameters
            for param in self.propagation.parameters():
                if param.grad is not None:
                    # Note we clip gradients here
                    grad = torch.clamp(param.grad.data, min=-5, max=5)

                    # SGD update
                    param.data -= self.args.inner_learning_rate * grad

    def fast_weights(self, x_list, x_domain_list, y_list, y_domain_list, names, scars):
        # Save a copy of the state dict before the updates
        optim = self.optimizers()
        
        weights_before = deepcopy(self.state_dict())

        # Performing the gradient adaptation over each label set
        grads, likelihoods, preds, signals = [], [], [], []
        for scar in torch.unique(scars):
            indices = torch.where(scars == scar)[0]
            # print(torch.unique(names), scar, indices, scar)

            sub_x = torch.stack([x_list[i] for i in indices])
            sub_xD = torch.stack([x_domain_list[i] for i in indices])
            sub_y = torch.stack([y_list[i] for i in indices])
            sub_yD = torch.stack([y_domain_list[i] for i in indices])
            name = torch.unique(names[indices])
            # print(sub_x.shape, sub_xD.shape, sub_y.shape, sub_yD.shape)

            # Reconstruct nodes based on subset size
            for data_idx, data_name in enumerate(self.args.data_names):
                self.construct_nodes(data_idx, data_name, 'data/ep/', sub_x.shape[0], None, self.args.devices[0], self.args.load_torso, self.args.load_physics, self.args.graph_method)

            """ Inner step """
            for _ in range(self.args.inner_steps):
                pred, _ = self(sub_x, sub_xD, sub_y, sub_yD, int(name))
                
                # Likelihood
                nll_raw = self.reconstruction_loss(pred, sub_x)
                nll_0 = nll_raw[:, :, 0].sum()
                nll_r = nll_raw[:, :, 1:].sum() / (sub_x.shape[-1] - 1)
                loss = sub_x.shape[-1] * (nll_0 * 0.1 + nll_r)
                
                # Get the loss terms from the specific latent dynamics loss
                model_specific_loss = self.model_specific_loss(sub_x, sub_xD, pred)
                likelihood = loss + model_specific_loss                

                # Get loss and gradients
                optim.zero_grad(set_to_none=True)
                self.manual_backward(likelihood)

                # Perform SGD over the dynamics function parameters
                for param in self.propagation.parameters():
                    if param.grad is not None:
                        # Note we clip gradients here
                        grad = torch.clamp(param.grad.data, min=-5, max=5)

                        # SGD update
                        param.data -= self.args.inner_learning_rate * grad

            self.embeddings = torch.concatenate([p.flatten() for p in self.propagation.parameters()]).unsqueeze(0).repeat(sub_x.shape[0], 1)

            """ Outer step """
            optim.zero_grad(set_to_none=True)
            pred, _ = self(sub_x, sub_xD, sub_y, sub_yD, int(name))
            
            # Likelihood
            nll_raw = self.reconstruction_loss(pred, sub_x)
            nll_0 = nll_raw[:, :, 0].sum()
            nll_r = nll_raw[:, :, 1:].sum() / (sub_x.shape[-1] - 1)
            loss = sub_x.shape[-1] * (nll_0 * 0.1 + nll_r)
            
            # Get the loss terms from the specific latent dynamics loss
            model_specific_loss = self.model_specific_loss(sub_x, sub_xD, pred)
            likelihood = loss + model_specific_loss         

            # Append and get gradients
            likelihoods.append(likelihood)
            preds.append(pred)
            signals.append(sub_x)
            self.manual_backward(likelihood)

            # Save grads
            sub_grads = []
            for p in self.parameters():
                if p.grad is not None:
                    sub_grads.append(torch.clamp(p.grad.data, min=-5, max=5))
                else:
                    sub_grads.append(None)

            grads.append(sub_grads)

            # Reload base weights
            self.load_state_dict(weights_before)

        # Pad to longest vertice length
        max_x_vertice = max([sub_x.shape[1] for sub_x in preds])
        for idx, (pq, xq) in enumerate(zip(preds, signals)):
            if max_x_vertice - xq.shape[1] > 0:
                preds[idx] = torch.nn.functional.pad(pq, pad=[0, 0, 0, max_x_vertice - pq.shape[1], 0, 0], mode='constant', value=0)
                signals[idx] = torch.nn.functional.pad(xq, pad=[0, 0, 0, max_x_vertice - xq.shape[1], 0, 0], mode='constant', value=0)
            
        return grads, likelihoods, torch.vstack(preds), torch.vstack(signals)

    def training_step(self, batch, batch_idx):
        """ Training step, getting loss and returning it to optimizer """
        # Get batch
        x, x_domain, y, y_domain, names, labels, scars = batch 
        scars = scars.unsqueeze(1)
 
        # Changeable k_shot
        k_shot = self.args.domain_size
        if self.args.domain_varying is True:
            k_shot = np.random.randint(1, x_domain.shape[1])
            x_domain = x_domain[:, :k_shot]
            y_domain = y_domain[:, :k_shot]

        # Turn into a list of individual tensors
        x_list = [xi for xi in x]
        x_domain_list = [xi for xi in x_domain]
        y_list = [yi for yi in y]
        y_domain_list = [yi for yi in y_domain]
        
        # Get memory batch, added only for training
        if self.memory is not None and self.task_counter > 0:
            memory_x, memory_x_domains, memory_y, memory_y_domains, memory_names, memory_scars = self.memory.get_batch(k_shot)
            
            # Turn into a list of individual tensors
            x_list = [xi for xi in x[:max(self.args.batch_size // 2, 2)]] + memory_x
            x_domain_list = [xi for xi in x_domain[:max(self.args.batch_size // 2, 2)]] + memory_x_domains
            y_list = [yi for yi in y[:max(self.args.batch_size // 2, 2)]] + memory_y
            y_domain_list = [yi for yi in y_domain[:max(self.args.batch_size // 2, 2)]] + memory_y_domains
            names = torch.vstack((names[:max(self.args.batch_size // 2, 2)], memory_names))
            scars = torch.vstack((scars[:max(self.args.batch_size // 2, 2)], memory_scars))
    
        """ All sample loop """
        grads, likelihoods, preds, signals = self.fast_weights(x_list, x_domain_list, y_list, y_domain_list, names, scars)

        """ Outer loop """
        optim = self.optimizers()
        optim.zero_grad()

        # Average across the samples
        likelihood = sum(likelihoods) / len(likelihoods)
        self.log_dict({"likelihood": likelihood}, prog_bar=True)

        # Aggregate grads
        agg_grads = [[] for _ in range(len(grads[0]))]
        for grad in grads:
            for sub_idx, sub_grad in enumerate(grad):
                agg_grads[sub_idx].append(sub_grad)

        for grad_idx, grad in enumerate(agg_grads):
            if grad[0] is None:
                agg_grads[grad_idx] = None
            else:
                agg_grads[grad_idx] = torch.stack(grad).mean([0])

        # Reapply grads
        for param_old, grad in zip(self.parameters(), agg_grads):
            if grad is not None:
                param_old.data.grad = grad
                param_old.grad = grad

        # Take the optimizer step and scheduler step
        optim.step()
        self.lr_schedulers().step()

        # Update memory with current batch
        if self.memory is not None:
            self.memory.batch_update(x[:self.args.batch_size // 2], y[:self.args.batch_size // 2], names[:self.args.batch_size // 2], scars[:self.args.batch_size // 2], self.task_counter + 1)

        # Return outputs as dict
        self.n_updates += 1
        self.task_steps += 1
        self.outputs.append({"preds": preds.detach().cpu(), "signals": signals.detach().cpu(), "names": names.detach().cpu()})
        return {"loss": likelihood}

    def test_step(self, batch, batch_idx):
        """ PyTorch-Lightning testing step """
        # Get model outputs from batch
        with torch.inference_mode(False):
            with torch.enable_grad():
                # Get batch
                x, x_domain, y, y_domain, names, labels, scars = batch 
        
                # Changeable k_shot
                k_shot = self.args.domain_size
                if self.args.domain_varying is True:
                    k_shot = np.random.randint(1, x_domain.shape[1])
                    x_domain = x_domain[:, :k_shot]
                    y_domain = y_domain[:, :k_shot]

                # Turn into a list of individual tensors
                x_list = [xi for xi in x]
                x_domain_list = [xi for xi in x_domain]
                y_list = [yi for yi in y]
                y_domain_list = [yi for yi in y_domain]
                
                """ All sample loop """
                _, _, preds, signals = self.fast_weights(x_list, x_domain_list, y_list, y_domain_list, names, scars)

            # Return output dictionary
            out = dict()
            for key, item in zip(["preds", "signals"], [preds, signals]):
                out[key] = item.detach().cpu().numpy()

            # Add meta-embeddings if it is a meta-model
            if self.args.meta is True:
                print(self.embeddings.shape)
                out["embeddings"] = self.embeddings.detach().cpu().numpy()

            return out
