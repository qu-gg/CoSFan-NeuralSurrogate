import torch
import functorch
import torch.nn as nn

from models.CommonComponents import get_params, one_hot_label, SpatialEncoder
from models.CommonTraining import LatentMetaDynamicsModel


class Transition_Recurrent_NoDomain(nn.Module):
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
        _g_t = self.act(self.lin1(z_t_1))
        g_t = self.act_weight(self.lin2(_g_t))
        _h_t = self.act(self.lin3(z_t_1))
        h_t = self.act(self.lin4(_h_t))
        _mu = self.lin_m(z_t_1)
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


class PNS(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)
        
        # encoder
        self.signal_encoder = SpatialEncoder(args.num_filters, args.latent_dim, cond=True)

        # Transition function
        self.propagation = Transition_Recurrent_NoDomain(args)

        # initialization
        self.condition_encoder = SpatialEncoder(args.num_filters, args.latent_dim)
        self.initial = nn.Linear(args.latent_dim, args.latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def construct_nodes(self, data_idx, heart_name, data_path, batch_size, device, load_torso, load_physics, graph_method):
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