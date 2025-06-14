import os
import torch
import numpy as np
import scipy.io
import pickle
import numbers
import itertools
from torch import nn
import torch.nn.init as weight_init
from torch.nn import functional as F
from torchdiffeq import odeint

from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv.spline_conv import SplineConv
from torch.utils.data import Dataset

from torch_geometric.data import Data


class HeartEmptyGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. The features and target values are 
    set to zeros in given graph.
    Not suitable for training.
    """

    def __init__(self,
                 mesh_graph,
                 label_type=None):
        self.graph = mesh_graph
        dim = self.graph.pos.shape[0]
        self.datax = np.zeros((dim, 101))
        self.label = np.zeros((101))

    def __len__(self):
        return (self.datax.shape[1])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.datax[:, [idx]]).float()  # torch.tensor(dataset[:,[i]],dtype=torch.float)
        y = torch.from_numpy(self.label[[idx]]).float()  # torch.tensor(label_aha[[i]],dtype=torch.float)

        sample = Data(x=x,
                      y=y,
                      edge_index=self.graph.edge_index,
                      edge_attr=self.graph.edge_attr,
                      pos=self.graph.pos)
        return sample


class Spatial_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 process,
                 stride=1,
                 padding=0,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None,
                 last_layer=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate = sample_rate
        self.last_layer = last_layer

        self.glayer = SplineConv(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0])

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            ),
            nn.ELU(inplace=True)
        )
    
    def forward(self, x, edge_index, edge_attr):
        N, V, C, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        res = self.residual(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, C)
        edge_index, edge_attr = expand(N, V, T, x.device, edge_index, edge_attr, self.sample_rate)
        x = F.elu(self.glayer(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        
        if self.last_layer is False:
            x = F.elu(x, inplace=True)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ST_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_seq,
                 out_seq,
                 dim,
                 kernel_size,
                 process,
                 stride=1,
                 padding=0,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 sample_rate=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_rate = sample_rate

        self.gcn = SplineConv(in_channels=in_channels, out_channels=out_channels, dim=dim, kernel_size=kernel_size[0])

        if process == 'e':
            self.tcn = nn.Sequential(
                nn.Conv2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        elif process == 'd':
            self.tcn = nn.Sequential(
                nn.ConvTranspose2d(
                    in_seq,
                    out_seq,
                    kernel_size[1],
                    stride,
                    padding
                ),
                nn.ELU(inplace=True)
            )
        else:
            raise NotImplementedError

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            ),
            nn.ELU(inplace=True)
        )

    def forward(self, x, edge_index, edge_attr):
        N, V, C, T = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        res = self.residual(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x.permute(3, 0, 1, 2).contiguous()
        x = x.view(-1, C)
        edge_index, edge_attr = expand(N, V, T, x.device, edge_index, edge_attr, self.sample_rate)
        x = F.elu(self.gcn(x, edge_index, edge_attr), inplace=True)
        x = x.view(T, N, V, -1)
        x = x.permute(1, 3, 0, 2).contiguous()

        x = x + res
        x = F.elu(x, inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = self.tcn(x)
        return x.permute(0, 3, 2, 1).contiguous()


class SpatialEncoder(nn.Module):
    def __init__(self, nf, latent_dim, cond=False):
        super().__init__()
        self.nf = nf
        self.latent_dim = latent_dim
        self.cond = cond

        if self.cond:
            self.conv1 = Spatial_Block(self.nf[0] * 2, self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        else:
            self.conv1 = Spatial_Block(self.nf[0], self.nf[1], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv2 = Spatial_Block(self.nf[1], self.nf[2], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv3 = Spatial_Block(self.nf[2], self.nf[3], dim=3, kernel_size=(3, 1), process='e', norm=False)
        self.conv4 = Spatial_Block(self.nf[3], self.nf[4], dim=3, kernel_size=(3, 1), process='e', norm=False)

        self.fce1 = nn.Conv2d(self.nf[4], self.nf[5], 1)
        self.fce2 = nn.Conv2d(self.nf[5], latent_dim, 1)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P01 = dict()
        self.P12 = dict()
        self.P23 = dict()
        self.P34 = dict()
    
    def setup_nodes(self, heart_idx, params):
        self.bg[heart_idx] = params["bg"]
        self.bg1[heart_idx] = params["bg1"]
        self.bg2[heart_idx] = params["bg2"]
        self.bg3[heart_idx] = params["bg3"]
        self.bg4[heart_idx] = params["bg4"]

        self.P01[heart_idx] = params["P01"]
        self.P12[heart_idx] = params["P12"]
        self.P23[heart_idx] = params["P23"]
        self.P34[heart_idx] = params["P34"]
    
    def forward(self, x, heart_name, y=None):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        # layer 1 (graph setup, conv, nonlinear, pool)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[0], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        if self.cond:
            y = y.view(batch_size, -1, self.nf[0], seq_len)
            x = torch.cat([x, y], dim=2)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P01[heart_name], x)
        
        # layer 2
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P12[heart_name], x)
        
        # layer 3
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.conv3(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P23[heart_name], x)

        # layer 4
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.conv4(x, edge_index, edge_attr)
        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P34[heart_name], x)

        # latent
        x = x.view(batch_size, -1, self.nf[4], seq_len)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = F.elu(self.fce1(x), inplace=True)
        x = torch.tanh(self.fce2(x))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class SpatialDecoder(nn.Module):
    def __init__(self, nf, latent_dim):
        super().__init__()
        self.nf = nf
        self.latent_dim = latent_dim

        self.fcd3 = nn.Conv2d(latent_dim, self.nf[5], 1)
        self.fcd4 = nn.Conv2d(self.nf[5], self.nf[4], 1)

        self.deconv4 = Spatial_Block(self.nf[4], self.nf[3], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv3 = Spatial_Block(self.nf[3], self.nf[2], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv2 = Spatial_Block(self.nf[2], self.nf[1], dim=3, kernel_size=(3, 1), process='d', norm=False)
        self.deconv1 = Spatial_Block(self.nf[1], self.nf[0], dim=3, kernel_size=(3, 1), process='d', norm=False)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

        self.P10 = dict()
        self.P21 = dict()
        self.P32 = dict()
        self.P43 = dict()
    
    def setup_nodes(self, heart_idx, params):
        self.bg[heart_idx] = params["bg"]
        self.bg1[heart_idx] = params["bg1"]
        self.bg2[heart_idx] = params["bg2"]
        self.bg3[heart_idx] = params["bg3"]
        self.bg4[heart_idx] = params["bg4"]
        
        self.P10[heart_idx] = params["P10"]
        self.P21[heart_idx] = params["P21"]
        self.P32[heart_idx] = params["P32"]
        self.P43[heart_idx] = params["P43"]
    
    def forward(self, x, heart_name):
        batch_size, seq_len = x.shape[0], x.shape[-1]
        x = x.permute(0, 2, 1, 3).contiguous()

        x = F.elu(self.fcd3(x), inplace=True)
        x = F.elu(self.fcd4(x), inplace=True)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.nf[4] * seq_len)
        x = torch.matmul(self.P43[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[4], seq_len), self.bg3[heart_name].edge_index, self.bg3[heart_name].edge_attr
        x = self.deconv4(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[3] * seq_len)
        x = torch.matmul(self.P32[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[3], seq_len), self.bg2[heart_name].edge_index, self.bg2[heart_name].edge_attr
        x = self.deconv3(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[2] * seq_len)
        x = torch.matmul(self.P21[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[2], seq_len), self.bg1[heart_name].edge_index, self.bg1[heart_name].edge_attr
        x = self.deconv2(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, self.nf[1] * seq_len)
        x = torch.matmul(self.P10[heart_name], x)
        x, edge_index, edge_attr = \
            x.view(batch_size, -1, self.nf[1], seq_len), self.bg[heart_name].edge_index, self.bg[heart_name].edge_attr
        x = self.deconv1(x, edge_index, edge_attr)

        x = x.view(batch_size, -1, seq_len)
        
        return x


class Transition_Recurrent(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.z_dim = args.latent_dim
        self.transition_dim = args.num_hidden
        self.identity_init = True
        self.stochastic = False

        # compute the gain (gate) of non-linearity
        self.lin1 = nn.Linear(self.z_dim*2, self.transition_dim*2)
        self.lin2 = nn.Linear(self.transition_dim*2, self.z_dim)
        # compute the proposed mean
        self.lin3 = nn.Linear(self.z_dim*2, self.transition_dim*2)
        self.lin4 = nn.Linear(self.transition_dim*2, self.z_dim)
        # compute the linearity part
        self.lin_m = nn.Linear(self.z_dim*2, self.z_dim)
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
    
    def forward(self, t, z_t_1, z_domain):
        z_combine = torch.cat((z_t_1, z_domain), dim=2)
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


class Transition_ODE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.latent_dim = args.latent_dim
        self.transition_dim = args.latent_dim
        self.num_layers = args.num_layers
        self.act_func = 'swish'
      
        self.combine = nn.Linear(2 *  self.latent_dim,  self.latent_dim)
        self.layers_dim = [2 *  self.latent_dim] + self.num_layers * [self.transition_dim] + [self.latent_dim]

        self.layers, self.acts = [], []
        for i, (n_in, n_out) in enumerate(zip(self.layers_dim[:-1], self.layers_dim[1:])):
            self.acts.append(get_act(self.act_func) if i < self.num_layers else get_act('tanh'))
            self.layers.append(nn.Linear(n_in, n_out, device=args.devices[0]))
        
    def ode_solver(self, t, x):
        x = torch.cat([x, self.embedding], dim=2)
        for a, layer in zip(self.acts, self.layers):
            x = a(layer(x))
        return x
    
    def forward(self, T, z_0, z_c=None):
        B = z_0.shape[0]
        t = torch.linspace(0, T - 1, T).to(self.args.devices[0])
        solver = lambda t, x: self.ode_solver(t, x)
    
        self.embedding = z_c
        
        zt = odeint(solver, z_0, t, method='rk4', options={'step_size': 0.25})
        return zt


class GCGRUCell(nn.Module):
    def __init__(self,
                 args,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.xr = SplineConv(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                        #  norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hr = SplineConv(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                        #  norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.xz = SplineConv(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                        #  norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hz = SplineConv(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                        #  norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.xn = SplineConv(in_channels=self.input_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                        #  norm=norm,
                         root_weight=root_weight,
                         bias=bias)

        self.hn = SplineConv(in_channels=self.hidden_dim,
                         out_channels=self.hidden_dim,
                         dim=dim,
                         kernel_size=self.kernel_size,
                         is_open_spline=is_open_spline,
                         degree=degree,
                        #  norm=norm,
                         root_weight=root_weight,
                         bias=bias)

    def forward(self, x, hidden, edge_index, edge_attr):
        r = torch.sigmoid(self.xr(x, edge_index, edge_attr) + self.hr(hidden, edge_index, edge_attr))
        z = torch.sigmoid(self.xz(x, edge_index, edge_attr) + self.hz(hidden, edge_index, edge_attr))
        n = torch.tanh(self.xn(x, edge_index, edge_attr) + r * self.hr(hidden, edge_index, edge_attr))
        h_new = (1 - z) * n + z * hidden
        return h_new

    def init_hidden(self, batch_size, graph_size):
        return torch.zeros(batch_size * graph_size, self.hidden_dim, device=self.args.devices[0])


class GCGRU(nn.Module):
    def __init__(self,
                 args,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dim,
                 is_open_spline=True,
                 degree=1,
                 norm=True,
                 root_weight=True,
                 bias=True,
                 num_layers=1,
                 return_all_layers=True):
        super().__init__()
        self.args = args
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(GCGRUCell(
                self.args,
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dim=dim,
                is_open_spline=is_open_spline,
                degree=degree,
                norm=norm,
                root_weight=root_weight,
                bias=bias
            ))
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None, edge_index=None, edge_attr=None):
        batch_size, graph_size, seq_len = x.shape[0], x.shape[1], x.shape[-1]

        if hidden_state is not None:
            raise NotImplemented
        else:
            hidden_state = self._init_hidden(batch_size=batch_size, graph_size=graph_size)
        
        layer_output_list = []
        last_state_list = []

        cur_layer_input = x.contiguous()
        for i in range(self.num_layers):
            h = hidden_state[i]
            output_inner = []
            for j in range(seq_len):
                cur = cur_layer_input[:, :, :, j].view(batch_size * graph_size, -1)
                h = h.view(batch_size * graph_size, -1)
                h = self.cell_list[i](
                    x=cur,
                    hidden=h,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )
                h = h.view(1, batch_size, graph_size, -1)
                output_inner.append(h)
            layer_output = torch.cat(output_inner, dim=0)
            layer_output = layer_output.permute(1, 2, 3, 0).contiguous()
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append(h)
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, graph_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, graph_size))
        return init_states


class RnnEncoder(nn.Module):
    def __init__(self, args, input_dim, rnn_dim, kernel_size, dim, is_open_spline=True, degree=1, norm=True,
                 root_weight=True, bias=True, n_layer=1, bd=True, orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.bd = bd
        
        self.rnn = GCGRU(
            args, 
            input_dim=input_dim,
            hidden_dim=rnn_dim,
            kernel_size=kernel_size,
            dim=dim,
            is_open_spline=is_open_spline,
            degree=degree,
            norm=norm,
            root_weight=root_weight,
            bias=bias,
            num_layers=n_layer
        )
        
        if orthogonal_init:
            self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters():
            if w.dim() > 1:
                weight_init.orthogonal_(w)
    
    def forward(self, x, edge_index, edge_attr):
        B, V, _, T = x.shape
        # seq_lengths = T * torch.ones(B).int().to(device)

        x = x.contiguous()
        hidden, _ = self.rnn(x, edge_index=edge_index, edge_attr=edge_attr)
        hidden = hidden[0]
        
        return hidden


class Aggregator(nn.Module):
    def __init__(self, rnn_dim, z_dim, time_dim, identity_init=True, stochastic=False):
        super().__init__()
        self.rnn_dim = rnn_dim
        self.z_dim = z_dim
        self.time_dim = time_dim
        self.stochastic = stochastic
        
        self.lin1 = nn.Linear(time_dim, 1)
        self.act = nn.ELU()

        self.lin2 = nn.Linear(rnn_dim, z_dim)
        self.lin_m = nn.Linear(z_dim, z_dim)
        self.lin_v = nn.Linear(z_dim, z_dim)
        self.act_v = nn.Tanh()

        if identity_init:
            self.lin_m.weight.data = torch.eye(z_dim)
            self.lin_m.bias.data = torch.zeros(z_dim)

    def forward(self, x):
        B, V, C, T = x.shape
        x = x.view(B, V * C, T)
        x = self.act(self.lin1(x))
        x = torch.squeeze(x)
        x = x.view(B, V, C)
        
        _mu = self.lin2(x)
        mu = self.lin_m(_mu)
        
        if self.stochastic:
            _var = self.lin_v(_mu)
            var = self.act_v(_var)
            return mu, var
        else:
            return mu


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "elu":
        return nn.ELU()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == "swish":
        return nn.SiLU()
    else:
        return None


def expand(batch_size, num_nodes, T, device, edge_index, edge_attr, sample_rate=None):
    num_edges = int(edge_index.shape[1] / batch_size)
    edge_index = edge_index[:, 0:num_edges]
    edge_attr = edge_attr[0:num_edges, :]


    sample_number = int(sample_rate * num_edges) if sample_rate is not None else num_edges
    selected_edges = torch.zeros(edge_index.shape[0], batch_size * T * sample_number).to(device)
    selected_attrs = torch.zeros(batch_size * T * sample_number, edge_attr.shape[1]).to(device)

    for i in range(batch_size * T):
        chunk = edge_index + num_nodes * i
        if sample_rate is not None:
            index = np.random.choice(num_edges, sample_number, replace=False)
            index = np.sort(index)
        else:
            index = np.arange(num_edges)
        selected_edges[:, sample_number * i:sample_number * (i + 1)] = chunk[:, index]
        selected_attrs[sample_number * i:sample_number * (i + 1), :] = edge_attr[index, :]

    selected_edges = selected_edges.long()
    return selected_edges, selected_attrs


def one_hot_label(label, N, V, T, device):
    y = torch.zeros([N, V, T]).to(device)
    for i, index in enumerate(label):
        y[i, index, :] = 1
    return y


def repeat(src, length):
    if isinstance(src, numbers.Number):
        src = list(itertools.repeat(src, length))
    return src


def node_degree(index, num_nodes=None, dtype=None, device=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))


def load_graph(filename, load_torso=0, graph_method=None):
    with open(filename + '.pickle', 'rb') as f:
        g = pickle.load(f)
        g1 = pickle.load(f)
        g2 = pickle.load(f)
        g3 = pickle.load(f)
        g4 = pickle.load(f)

        P10 = pickle.load(f)
        P21 = pickle.load(f)
        P32 = pickle.load(f)
        P43 = pickle.load(f)

        if load_torso == 1:
            t_g = pickle.load(f)
            t_g1 = pickle.load(f)
            t_g2 = pickle.load(f)
            t_g3 = pickle.load(f)

            t_P10 = pickle.load(f)
            t_P21 = pickle.load(f)
            t_P32 = pickle.load(f)

            if graph_method == 'bipartite':
                Hs = pickle.load(f)
                Ps = pickle.load(f)
            else:
                raise NotImplementedError

    if load_torso == 0:
        P01 = P10 / P10.sum(axis=0)
        P12 = P21 / P21.sum(axis=0)
        P23 = P32 / P32.sum(axis=0)
        P34 = P43 / P43.sum(axis=0)

        P01 = torch.from_numpy(np.transpose(P01)).float()
        P12 = torch.from_numpy(np.transpose(P12)).float()
        P23 = torch.from_numpy(np.transpose(P23)).float()
        P34 = torch.from_numpy(np.transpose(P34)).float()

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34
    elif load_torso == 1:
        t_P01 = t_P10 / t_P10.sum(axis=0)
        t_P12 = t_P21 / t_P21.sum(axis=0)
        t_P23 = t_P32 / t_P32.sum(axis=0)

        t_P01 = torch.from_numpy(np.transpose(t_P01)).float()
        t_P12 = torch.from_numpy(np.transpose(t_P12)).float()
        t_P23 = torch.from_numpy(np.transpose(t_P23)).float()

        if graph_method == 'bipartite':
            Ps = torch.from_numpy(Ps).float()
        else:
            raise NotImplementedError

        P10 = torch.from_numpy(P10).float()
        P21 = torch.from_numpy(P21).float()
        P32 = torch.from_numpy(P32).float()
        P43 = torch.from_numpy(P43).float()

        return g, g1, g2, g3, g4, P10, P21, P32, P43,\
            t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps


def get_params(data_path, heart_name, device, batch_size, load_torso=0, load_physics=0, graph_method=None):
    if load_physics == 1:
        # Load physics parameters
        physics_name = heart_name.split('_')[0]
        physics_dir = os.path.join(data_path, 'physics/{}/'.format(physics_name))
        mat_files = scipy.io.loadmat(os.path.join(physics_dir, 'h_L.mat'), squeeze_me=True, struct_as_record=False)
        L = mat_files['h_L']

        mat_files = scipy.io.loadmat(os.path.join(physics_dir, 'H.mat'), squeeze_me=True, struct_as_record=False)
        H = mat_files['H']

        L = torch.from_numpy(L).float().to(device)
        print('Load Laplacian: {} x {}'.format(L.shape[0], L.shape[1]))

        H = torch.from_numpy(H).float().to(device)
        print('Load H matrix: {} x {}'.format(H.shape[0], H.shape[1]))

    # Load geometrical parameters
    graph_file = os.path.join(data_path, 'signal/{}/{}_{}'.format(heart_name, heart_name, graph_method))
    if load_torso == 0:
        g, g1, g2, g3, g4, P10, P21, P32, P43, P01, P12, P23, P34 = \
            load_graph(graph_file, load_torso, graph_method)
    else:
        g, g1, g2, g3, g4, P10, P21, P32, P43,\
        t_g, t_g1, t_g2, t_g3, t_P01, t_P12, t_P23, Hs, Ps = load_graph(graph_file, load_torso, graph_method)

    num_nodes = [g.pos.shape[0], g1.pos.shape[0], g2.pos.shape[0], g3.pos.shape[0],
                 g4.pos.shape[0]]
    # print('number of nodes:', num_nodes)

    g_dataset = HeartEmptyGraphDataset(mesh_graph=g)
    g_loader = DataLoader(g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg = next(iter(g_loader))

    g1_dataset = HeartEmptyGraphDataset(mesh_graph=g1)
    g1_loader = DataLoader(g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg1 = next(iter(g1_loader))

    g2_dataset = HeartEmptyGraphDataset(mesh_graph=g2)
    g2_loader = DataLoader(g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg2 = next(iter(g2_loader))

    g3_dataset = HeartEmptyGraphDataset(mesh_graph=g3)
    g3_loader = DataLoader(g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg3 = next(iter(g3_loader))

    g4_dataset = HeartEmptyGraphDataset(mesh_graph=g4)
    g4_loader = DataLoader(g4_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bg4 = next(iter(g4_loader))

    P10 = P10.to(device)
    P21 = P21.to(device)
    P32 = P32.to(device)
    P43 = P43.to(device)

    bg1 = bg1.to(device)
    bg2 = bg2.to(device)
    bg3 = bg3.to(device)
    bg4 = bg4.to(device)

    bg = bg.to(device)

    if load_torso == 0:
        P01 = P01.to(device)
        P12 = P12.to(device)
        P23 = P23.to(device)
        P34 = P34.to(device)

        P1n = np.ones((num_nodes[1], 1))
        Pn1 = P1n / P1n.sum(axis=0)
        Pn1 = torch.from_numpy(np.transpose(Pn1)).float()
        P1n = torch.from_numpy(P1n).float()
        P1n = P1n.to(device)
        Pn1 = Pn1.to(device)

        params = {
            "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
            "P01": P01, "P12": P12, "P23": P23, "P34": P34,
            "P10": P10, "P21": P21, "P32": P32, "P43": P43,
            "P1n": P1n, "Pn1": Pn1, "num_nodes": num_nodes, "g": g, "bg": bg
        }
    elif load_torso == 1:
        t_num_nodes = [t_g.pos.shape[0], t_g1.pos.shape[0], t_g2.pos.shape[0], t_g3.pos.shape[0]]
        print('number of nodes on torso:', t_num_nodes)
        t_g_dataset = HeartEmptyGraphDataset(mesh_graph=t_g)
        t_g_loader = DataLoader(t_g_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg = next(iter(t_g_loader))

        t_g1_dataset = HeartEmptyGraphDataset(mesh_graph=t_g1)
        t_g1_loader = DataLoader(t_g1_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg1 = next(iter(t_g1_loader))

        t_g2_dataset = HeartEmptyGraphDataset(mesh_graph=t_g2)
        t_g2_loader = DataLoader(t_g2_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg2 = next(iter(t_g2_loader))

        t_g3_dataset = HeartEmptyGraphDataset(mesh_graph=t_g3)
        t_g3_loader = DataLoader(t_g3_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        t_bg3 = next(iter(t_g3_loader))

        t_P01 = t_P01.to(device)
        t_P12 = t_P12.to(device)
        t_P23 = t_P23.to(device)

        t_bg1 = t_bg1.to(device)
        t_bg2 = t_bg2.to(device)
        t_bg3 = t_bg3.to(device)
        t_bg = t_bg.to(device)

        if graph_method == 'bipartite':
            H_dataset = HeartEmptyGraphDataset(mesh_graph=Hs)
            H_loader = DataLoader(H_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            H_inv = next(iter(H_loader))

            H_inv = H_inv.to(device)
            Ps = Ps.to(device)

            if load_physics == 1:
                params = {
                    "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
                    "P10": P10, "P21": P21, "P32": P32, "P43": P43,
                    "num_nodes": num_nodes, "g": g, "bg": bg,
                    "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
                    "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
                    "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
                    "H_inv": H_inv, "P": Ps,
                    "H": H, "L": L
                }
            else:
                params = {
                    "bg1": bg1, "bg2": bg2, "bg3": bg3, "bg4": bg4,
                    "P10": P10, "P21": P21, "P32": P32, "P43": P43,
                    "num_nodes": num_nodes, "g": g, "bg": bg,
                    "t_bg1": t_bg1, "t_bg2": t_bg2, "t_bg3": t_bg3,
                    "t_P01": t_P01, "t_P12": t_P12, "t_P23": t_P23,
                    "t_num_nodes": t_num_nodes, "t_g": t_g, "t_bg": t_bg,
                    "H_inv": H_inv, "P": Ps
                }
        else:
            raise NotImplementedError

    return params
