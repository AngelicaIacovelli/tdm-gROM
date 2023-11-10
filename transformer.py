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
    while using single attention head.
    The implementation follows https://openreview.net/pdf?id=XctLdNfCmP
    """

    def __init__(self, N_t, N_lat, N_inn, N_g, N_mu, N_neu_MLP_p, N_hid_MLP_p, N_neu_MLP_m, N_hid_MLP_m):
        super(TransformerCell, self).__init__()

        self.W_1 = Linear(N_inn, N_lat, bias=False).float()
        self.W_2 = Linear(N_lat, N_inn, bias=False).float()
        self.W_3 = Linear(N_lat, N_g, bias=False).float()

        self.MLP_p = MLP(N_mu, N_lat, N_neu_MLP_p, N_hid_MLP_p, normalize=False) # True?
        self.MLP_m = MLP(N_g, N_lat, N_neu_MLP_m, N_hid_MLP_m, normalize=False)

        self.N_t = N_t
        self.N_lat = N_lat
        self.N_lat_sqrt = th.sqrt(th.tensor(self.N_lat, dtype=th.float32))

    def forward(self, mu, z_0):
        """
        Forward step

        Arguments:
            mu: vector of dimension N_t, containing flow rates at the inlet. 
            z_0: vector of dimension N_lat, containing the encoded initial condition.
        Returns:
            Z_tilde: matrix of dimension [N_t + 1, N_lat], containing all the processed encoded time steps.
        """

        Z_tilde = self.MLP_p(mu)
        Z_tilde = th.cat((Z_tilde, z_0), dim = 0)
        for idx_t in th.arange(2, self.N_t + 1):
            a = softmax(th.mv(self.W_1(self.W_2(Z_tilde)), Z_tilde[idx_t - 1, :]) / self.N_lat_sqrt, dim = 0)
            g = th.matmul(a, self.W_3(Z_tilde))
            Z_tilde = th.cat((Z_tilde, th.reshape(Z_tilde[idx_t - 1, :] + self.MLP_m(g), (1, self.N_lat))), dim = 0)

        return Z_tilde
