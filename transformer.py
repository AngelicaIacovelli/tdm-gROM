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


class TransformerCell(Module):
    """
    Transformer cell.

    This class computes the temporal evolution of a set of latent variables
<<<<<<< Updated upstream
    while using single attention head.
=======
    while using multi attention head.
>>>>>>> Stashed changes
    The implementation follows https://openreview.net/pdf?id=XctLdNfCmP
    """

    def __init__(self, cfg):
        super(TransformerCell, self).__init__()

<<<<<<< Updated upstream
=======
        self.N_heads = cfg.transformer_architecture.N_heads
>>>>>>> Stashed changes
        self.N_t = cfg.transformer_architecture.N_timesteps
        self.N_lat = cfg.architecture.latent_size_AE
        self.N_inn = cfg.transformer_architecture.N_inn
        self.N_g = cfg.transformer_architecture.N_g
        self.N_mu = cfg.transformer_architecture.N_timesteps
        self.N_neu_MLP_p = cfg.transformer_architecture.N_neu_MLP_p
        self.N_hid_MLP_p = cfg.transformer_architecture.N_hid_MLP_p
        self.N_neu_MLP_m = cfg.transformer_architecture.N_neu_MLP_m
        self.N_hid_MLP_m = cfg.transformer_architecture.N_hid_MLP_m

<<<<<<< Updated upstream
        self.W_1 = Linear(self.N_inn, self.N_lat, bias=False).float()
        self.W_2 = Linear(self.N_lat, self.N_inn, bias=False).float()
        self.W_3 = Linear(self.N_lat, self.N_g, bias=False).float()

        self.MLP_p = MLP(self.N_mu, self.N_lat, self.N_neu_MLP_p, self.N_hid_MLP_p, normalize=False) # True?
        self.MLP_m = MLP(self.N_g, self.N_lat, self.N_neu_MLP_m, self.N_hid_MLP_m, normalize=False)
=======
        self.W_1 = th.nn.ModuleList([Linear(self.N_inn, self.N_lat, bias=False).float() for _ in range(self.N_heads)])
        self.W_2 = th.nn.ModuleList([Linear(self.N_lat, self.N_inn, bias=False).float() for _ in range(self.N_heads)])
        self.W_3 = th.nn.ModuleList([Linear(self.N_lat, self.N_g, bias=False).float() for _ in range(self.N_heads)])

        self.MLP_p = MLP(self.N_mu, self.N_lat, self.N_neu_MLP_p, self.N_hid_MLP_p, normalize=False) # True?
        self.MLP_m = MLP(self.N_heads * self.N_g, self.N_lat, self.N_neu_MLP_m, self.N_hid_MLP_m, normalize=False)
>>>>>>> Stashed changes

        self.N_lat_sqrt = th.sqrt(th.tensor(self.N_lat, dtype=th.float32))

        self.device = "cuda" if th.cuda.is_available() else "cpu"

<<<<<<< Updated upstream
    def forward(self, mu, z_0):
=======
    def forward(self, mu, z_0, N_t):
>>>>>>> Stashed changes
        """
        Forward step

        Arguments:
            mu: matrix of dimension [N_batch, N_t], containing flow rates at the inlet. 
            z_0: matrix of dimension [N_batch, N_lat], containing the encoded initial condition.
<<<<<<< Updated upstream
=======
            N_t: number of timesteps for loss incremental training (N_t = self.N_t to train over the whole simulation).
>>>>>>> Stashed changes
        Returns:
            Z_tilde: tensor of dimension [N_batch, N_t + 1, N_lat], containing all the processed encoded time steps.
        """

        N_batch = z_0.shape[0]
        z_0 = z_0.unsqueeze(1)
        Z_tilde = self.MLP_p(mu)
        Z_tilde = Z_tilde.unsqueeze(1)
        Z_tilde = th.cat((Z_tilde, z_0), dim = 1)
<<<<<<< Updated upstream
        for idx_t in th.arange(2, self.N_t + 1):
            a = softmax(th.bmm(self.W_1(self.W_2(Z_tilde)), Z_tilde[:, idx_t - 1, :].unsqueeze(2)) / self.N_lat_sqrt, dim = 1) # dim=1?
            g = th.sum(th.mul(self.W_3(Z_tilde), a), dim = 1)
=======
        for idx_t in th.arange(2, N_t + 1):
            a = []
            g = []
            for idx_h in range(self.N_heads):
                a = softmax(th.bmm(self.W_1[idx_h](self.W_2[idx_h](Z_tilde)), Z_tilde[:, idx_t - 1, :].unsqueeze(2)) / self.N_lat_sqrt, dim = 1)
                if idx_h == 0:
                    g = th.sum(th.mul(self.W_3[idx_h](Z_tilde), a), dim = 1)
                else:
                    g = th.cat((g, th.sum(th.mul(self.W_3[idx_h](Z_tilde), a), dim = 1)), dim = 1)
>>>>>>> Stashed changes
            Z_tilde = th.cat((Z_tilde, (Z_tilde[:, idx_t - 1, :] + self.MLP_m(g)).unsqueeze(1)), dim = 1)

        return Z_tilde
