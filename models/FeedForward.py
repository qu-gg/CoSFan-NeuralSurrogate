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

    def construct_nodes(self, data_idx, heart_name, data_path, batch_size, k_shot, device, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, device, batch_size, load_torso, load_physics, graph_method)        
        self.condition_encoder.setup_nodes(data_idx, params)
        self.decoder.setup_nodes(data_idx, params)
        
        params = get_params(data_path, heart_name, device, batch_size * k_shot, load_torso, load_physics, graph_method)       
        self.signal_encoder.setup_nodes(data_idx, params)
        
        self.bg[data_idx] = params["bg"]
        self.bg1[data_idx] = params["bg1"]
        self.bg2[data_idx] = params["bg2"]
        self.bg3[data_idx] = params["bg3"]
        self.bg4[data_idx] = params["bg4"]

    def latent_initial(self, y, N, V, heart_name):
        y = one_hot_label(y[:, 2] - 1, N, V, 1, self.args.devices[0])
        y = y[:, :].view(N, V, 1)
        z_0 = self.condition_encoder(y, heart_name)
        z_0 = torch.squeeze(z_0)
        z_0 = self.initial(z_0)
        
        return z_0

    def latent_domain(self, D_x, D_y, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr
        print(edge_index.shape, edge_attr.shape)
        N, K, V, T = D_x.shape
        
        # Reshape D_x to process all K samples at once while maintaining batch structure
        # From [N, K, V, T] to [N*K, V, T]
        D_x = D_x.reshape(N*K, V, T)
        
        # Similarly reshape D_y and create one-hot labels
        D_y = D_y.reshape(N*K, -1)
        D_y = one_hot_label(D_y[:, 2] - 1, N*K, V, T, self.args.devices[0])
        
        # Process all samples through signal encoder at once
        z = self.signal_encoder(D_x, heart_name, D_y)
        z_c = self.domain_seq(z, edge_index, edge_attr)
        z_c = self.domain(z_c)
        
        # Reshape to the base dimensions
        z_c = z_c.reshape(N, K, z_c.shape[1], z_c.shape[2])
        
        # Derive just context-samples embeddings
        z_c_context = torch.mean(z_c[:, :-1], dim=1)  # Average over K dimension
        mu_c_context = self.mu_c(z_c_context)
        logvar_c_context = self.var_c(z_c_context)
        mu_c_context = torch.clamp(mu_c_context, min=-100, max=85)
        logvar_c_context = torch.clamp(logvar_c_context, min=-100, max=85)

        # Reparameterization
        std = torch.exp(0.5 * logvar_c_context)
        eps = torch.randn_like(std)
        z_c_context = mu_c_context + eps * std
        
        # Derive the KL based distributional parameters - that includes GT query samples 
        z_c_gt = torch.mean(z_c, dim=1)  # Average over K dimension
        mu_c_gt = self.mu_c(z_c_gt)
        logvar_c_gt = self.var_c(z_c_gt)
        self.mean_t = torch.clamp(mu_c_gt, min=-100, max=85)
        self.logvar_t = torch.clamp(logvar_c_gt, min=-100, max=85)
        
        return z_c_context, mu_c_context, logvar_c_context
    
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
        
        # Prepare full context set including current sample
        x_expanded = x.view(N, 1, V, T)
        y_expanded = y.view(N, 1, -1)
        xD = torch.cat([xD, x_expanded], dim=1)
        yD = torch.cat([yD, y_expanded], dim=1)
        
        # Domain encoding
        self.embeddings, self.mean_c, self.logvar_c = self.latent_domain(xD, yD, name)

        # Latent propagation
        z = self.time_modeling(T, z0, self.embeddings)
        
        # Decode
        x_ = self.decoder(z, name)    
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
