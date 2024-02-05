import os
import sys
import torch as th
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn.functional import softmax
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
import dgl

#from linear_multihead_attention import LinearMultiheadAttention

class MLP(Module):
    """
    Multi-layer perceptron.

    Attributes:
        input: Linear pytorch module
        output: Linear pytorch module
        n_h_layers (int): number of hidden layers
        hidden_layers: list of Linear modules
        normalize (bool): specifies if last LayerNorm should be applied after
                          last layer
        norm: LayerNorm pytorch module

    """

    def __init__(self, in_feats, out_feats, latent_space, n_h_layers, normalize=True):
        """
        Init MLP.

        Initialize MLP.

        Arguments:
            in_feats (int): number of input features
            out_feats (int): number of output features
            latent_space (int): size of the latent space
            n_h_layers (int): number of hidden layers
            normalize (bool): specifies whether normalization should be applied
                              in last layer. Default -> true

        """
        super().__init__()
        self.input = Linear(in_feats, latent_space, bias=True).float()
        self.output = Linear(latent_space, out_feats, bias=True).float()
        self.n_h_layers = n_h_layers
        self.hidden_layers = th.nn.ModuleList()
        for i in range(self.n_h_layers):
            self.hidden_layers.append(
                Linear(latent_space, latent_space, bias=True).float()
            )

        self.normalize = normalize
        if self.normalize:
            self.norm = LayerNorm(out_feats).float()

    def forward(self, inp):
        """
        Forward step

        Arguments:
            inp: input tensor

        Returns:
            result of forward step

        """
        f = self.input(inp)
        f = F.leaky_relu(f)

        for i in range(self.n_h_layers):
            f = self.hidden_layers[i](f)
            f = F.leaky_relu(f)

        # enc_features = self.dropout(enc_features)
        f = self.output(f)

        if self.normalize:
            f = self.norm(f)

        return f


class LNODECell(Module):
    """
    PyTorch cell representing Latent Neural Ordinary Differential Equations.

    N_states     -> Dimensionality of the z vector (number of physical variables/outputs + number of latent variables).
    N_parameters -> Number of input parameters.
    N_hid_MLP    -> Number of hidden layers of the feedforward fully connected neural network.
    N_neu_MLP    -> Number of neurons per hidden layer of the feedforward fully connected neural network.
      
    dt           -> Time step of the numerical simulation.

    params       -> matrix of dimension (number of parameters, number of time steps, number of simulations).
    z_0_physical -> matrix of dimension (number of physical variables, number of simulations)

    Z_tilde      -> matrix of dimension (number of outputs, number of time steps, number of simulations).
    
    Transformer: number of physical variables = number of outputs
    """

    def __init__(self, cfg, dt):
        super(LNODECell, self).__init__()

        self.N_states = cfg.LNODEs_architecture.N_states
        self.N_parameters = cfg.LNODEs_architecture.N_parameters
        self.N_hid_MLP = cfg.LNODEs_architecture.N_hid_MLP
        self.N_neu_MLP = cfg.LNODEs_architecture.N_neu_MLP

        self.dt = dt

        self.NN = MLP(self.N_states + self.N_parameters, self.N_states, self.N_neu_MLP, self.N_hid_MLP, normalize = False)

        self.device = "cuda" if th.cuda.is_available() else "cpu"

    def forward(self, params, z_0_physical):
        num_outs = z_0_physical.shape[0] 
        z_0_latent = th.zeros((self.N_states - num_outs, z_0_physical.shape[1]), device=self.device)
        Z_tilde = th.cat((z_0_physical, z_0_latent), dim = 0).unsqueeze(-1)

        for idx_t in range(params.shape[1] - 1):
            Z_tilde = th.cat((Z_tilde, (Z_tilde[:, :, -1] + self.dt * self.NN(th.cat((Z_tilde[:, :, -1], params[:, idx_t, :]), dim = 0).permute(1, 0)).permute(1, 0)).unsqueeze(-1)), dim = 2)

        return Z_tilde[:num_outs, :, :].permute(0, 2, 1)