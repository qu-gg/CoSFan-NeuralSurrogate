import torch
import functorch
import torch.nn as nn

from models.CommonComponents import RnnEncoder, Aggregator, get_params, one_hot_label, SpatialEncoder, SpatialDecoder
from models.CommonTraining import LatentMetaDynamicsModel



class FeedForward(LatentMetaDynamicsModel):
    def __init__(self, args):
        super().__init__(args)
        
        # encoder
        self.signal_encoder = SpatialEncoder(args.num_filters, args.latent_dim, cond=True)

        # Domain model
        self.domain_seq = RnnEncoder(args, 
                                     args.latent_dim, 
                                     args.latent_dim,
                                     dim=3,
                                     kernel_size=3,
                                     norm=False,
                                     n_layer=1,
                                     bd=False)
        self.domain = Aggregator(args.latent_dim, args.latent_dim, args.window - args.omit, stochastic=False)
        self.mu_c = nn.Linear(args.latent_dim, args.latent_dim)
        self.var_c = nn.Linear(args.latent_dim, args.latent_dim)

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

    def latent_domain(self, D_x, D_y, K, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        N, _, V, T = D_x.shape
        D_z_c = []
        for i in range(K):
            D_xi = D_x[:, i, :, :].view(N, V, T)
            D_yi = D_y[:, i, :]
            D_yi = one_hot_label(D_yi[:, 2] - 1, N, V, T, self.args.devices[0])
            z_i = self.signal_encoder(D_xi, heart_name, D_yi)
            z_c_i = self.domain_seq(z_i, edge_index, edge_attr)
            D_z_c.append(self.domain(z_c_i))

        z_c = sum(D_z_c) / len(D_z_c)
        mu_c = self.mu_c(z_c)
        logvar_c = self.var_c(z_c)
        mu_c = torch.clamp(mu_c, min=-100, max=85)
        logvar_c = torch.clamp(logvar_c, min=-100, max=85)

        # Reparameterization
        std = torch.exp(0.5 * logvar_c)
        eps = torch.randn_like(std)
        z_c =  mu_c + eps * std
        return z_c, mu_c, logvar_c
    
    def time_modeling(self, T, z_0, z_c):
        # print(z_0.shape)
        N, V, C = z_0.shape


        z_prev = z_0
        z = []
        for i in range(1, T):
            z_t = self.propagation(T, z_prev, z_c)
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
        
        # Domain encoding
        self.embeddings, self.mean_c, self.logvar_c = self.latent_domain(xD, yD, xD.shape[1], name)

        # Latent propagation
        z = self.time_modeling(T, z0, self.embeddings)
        
        # Decode
        x_ = self.decoder(z, name)    
        
        # KL p(c | D u x)
        x = x.view(N, 1, -1, T)
        y = y.view(N, 1, -1)
        D_x_cat = torch.cat([xD, x], dim=1)
        D_y_cat = torch.cat([yD, y], dim=1)
        _, self.mean_t, self.logvar_t = self.latent_domain(D_x_cat, D_y_cat, xD.shape[1], name)
        return x_, z
    
    def kl_div(self, mu1, var1, mu2=None, var2=None):
        if mu2 is None:
            mu2 = torch.zeros_like(mu1)
        if var2 is None:
            var2 = torch.zeros_like(mu1)

        return 0.5 * (var2 - var1 + (torch.exp(var1) + (mu1 - mu2).pow(2)) / torch.exp(var2) - 1)

    def kl_div_stn(self, mu, logvar):
        return 0.5 * (mu.pow(2) + torch.exp(logvar) - logvar - 1)

    def model_specific_loss(self, x, xD, names):
        # Ignore loss if it is a deterministic model
        if self.args.stochastic is False:
            return 0.0
        
        # KL on C with a prior of Normal
        kl_c_normal = 1e-4 * self.kl_div(self.mean_c, self.logvar_c).sum() / x.shape[0]
        self.log("kl_c_normal", kl_c_normal, prog_bar=True)

        # KL on C with a prior of the context set with itself in it
        kl_c_context = 0.1 * self.kl_div(self.mean_c, self.logvar_c, self.mean_t, self.logvar_t).sum() / x.shape[0]
        self.log("kl_c_context", kl_c_context, prog_bar=True)

        # Return them as one loss
        return  kl_c_normal + kl_c_context
