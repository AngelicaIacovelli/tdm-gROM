import os
import sys
#sys.path.append(os.getcwd())
import torch as th
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
#import graph1d.generate_normalized_graphs as nz
#import json

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
    def __init__(self, in_feats, out_feats, latent_space, n_h_layers,
                normalize = True):
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
        self.input = Linear(in_feats,latent_space,bias = True).float()
        self.output = Linear(latent_space, out_feats, bias = True).float()
        self.n_h_layers = n_h_layers
        self.hidden_layers = th.nn.ModuleList()
        for i in range(self.n_h_layers):
            self.hidden_layers.append(Linear(latent_space,
                                             latent_space,
                                             bias = True).float())

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

class GLSTMCell(Module):
    """
    Graph LSTM cell.

    This class computes pressure and flowrate updates given the previous system
    state.

    """

  def __init__(self,cfg):

    super(GLSTMCell, self).__init__()


    self.encoder_nodes = MLP( cfg.architecture.in_feats,
                              cfg.architecture.latent_size_gnn,  
                              cfg.architecture.latent_size_mlp,  
                              cfg.architecture.number_hidden_layers_mlp)
    self.encoder_edges = MLP( cfg.architecture.edge_feats,
                              cfg.architecture.latent_size_gnn,  
                              cfg.architecture.latent_size_mlp,  
                              cfg.architecture.number_hidden_layers_mlp)
    self.output = MLP(cfg.architecture.latent_size_gnn,
                      cfg.architecture.out_size,
                      cfg.architecture.latent_size_mlp,
                      cfg.architecture.number_hidden_layers_mlp,
                      False)

    hidden_dim_l = cfg.architecture.hidden_dim_l
    hidden_dim_h = cfg.architecture.hidden_dim_h
    in_feats = cfg.architecture.in_feats
    edge_feats = cfg.architecture.edge_feats

    self.W_i = Linear(in_feats, hidden_dim_l, bias = True).float()
    self.U_i = Linear(hidden_dim_h,hidden_dim_l, bias = False).float()

    self.W_f = Linear(edge_feats, hidden_dim_l, bias = True).float()
    self.U_f = Linear(hidden_dim_h,hidden_dim_l, bias = False).float()

    self.W_o = Linear(in_feats, hidden_dim_l, bias = True).float()
    self.U_o = Linear(hidden_dim_h,hidden_dim_l, bias = False).float()

    self.W_u = Linear(in_feats, hidden_dim_l, bias = True).float()
    self.U_u = Linear(hidden_dim_h,hidden_dim_l, bias = False).float()


  def encode_nodes(self, nodes):
    """
    Encode graph nodes

    Arguments:
        edges: graph nodes

    Returns:
        dictionary (key: 'proc_nodes', value: encoded features)

    """
    features = nodes.data['nfeatures_w_bcs']
    enc_features = self.encoder_nodes(features)
    return {'proc_node': enc_features}


  def encode_edges(self, edges):
    """
    Encode graph edges

    Arguments:
        edges: graph edges

    Returns:
        dictionary (key: 'proc_edge', value: encoded features)

    """
    enc_features = self.encoder_edges(edges.data['efeatures'])
    return {'proc_edge': enc_features}


  def decode_nodes(self, nodes):
    """
    Decode graph nodes

    Arguments:
        nodes: graph nodes

    Returns:
        dictionary (key: 'pred_labels', value: decoded features)

    """
    h = self.output(nodes.data['proc_node'])
    return {'pred_labels': h}


  def continuity_loss(self, g, flowrate, take_mean = True):
    """
    Compute contiuity loss

    Continuity loss as the mass loss occurring  at junctions.

    Arguments:
        g: graph
        flowrate: tensor containing nodal values of flowrate
        take_mean: if True, take mean of junction losses. If
                    False, take sum. Default -> True.
    Returns:
        sum of mass loss occurring at branches and at junctions

    """
    g.ndata['next_flowrate'] = flowrate.clone()

    # we zero-out inlet and outlet flowrate (otherwise they would send
    # their flowrate to branch and junction nodes)
    g.ndata['next_flowrate'][g.ndata['inlet_mask'].bool()] = 0
    g.ndata['next_flowrate'][g.ndata['outlet_mask'].bool()] = 0

    # # we send flowrate through branches, compute the mean
    # # of neighboring nodes, and compute the diff with our estimate
    # g.update_all(fn.copy_u('next_flowrate', 'm'),
    #              fn.sum('m', 'sum_flowrate'))
    # # branch nodes have only two neighbors
    # diff = th.abs(2 * g.ndata['next_flowrate'] - g.ndata['sum_flowrate'])
    # diff = diff * g.ndata['continuity_mask']
    # if take_mean:
    #     branch_continuity = th.mean(diff)
    # else:
    #     branch_continuity = th.sum(diff)

    # we keep flowrate at inlet and outlets of junctions
    g.ndata['flow_junction'] = g.ndata['next_flowrate'] * \
                                g.ndata['jun_mask']

    g.update_all(fn.copy_u('flow_junction', 'm'),
                  fn.sum('m', 'sum_flowrate'))

    # we use the inlet to compute the difference
    diff = th.abs(g.ndata['sum_flowrate'] - g.ndata['next_flowrate'])
    diff = diff * g.ndata['jun_inlet_mask']

    if take_mean:
        junction_continuity = th.sum(diff) / \
                            th.sum(g.ndata['jun_inlet_mask'])
    else:
        junction_continuity = th.sum(diff)

    return junction_continuity


#VECTOR i

  def compute_Uih(self, edges):
    weight_matrix_U = self.U_i.weight
    #print("This is U_i: ", weight_matrix_U)

    h = edges.src['h']
    Uih = self.U_i(h)
    #print("This is Ui*h: ", Uih)
    return {'Uih': Uih}

  def compute_i(self,nodes):
    x = nodes.data['proc_node'] #dictionary (key: 'proc_node', value: processed features)

    Uih_sum = nodes.data['Uih_sum']
    #print("This is Uih_sum, the sum of the neighbors' Uih per each node: ", Uih_sum)

    weight_matrix_W = self.W_i.weight
    bias = self.W_i.bias
    #print("This is W_i: ", weight_matrix_W)
    #print("This is the bias of W: ",bias)

    Wx = self.W_i(x)
    #print("This is Wx: ", Wx)

    i = Wx + Uih_sum
    #print("This is i: ", i)

    i = th.nn.functional.leaky_relu(i)
    #print("This is leaky_relu(i): ", i)

    return {'i': i}


# VECTOR f

  def compute_f(self, edges):
    x = edges.data['efeatures']
    x = x.float()
    Wx = self.W_f(x)

    h = edges.src['h']
    Ufh = self.U_f(h)

    f = Wx + Ufh
    f = th.nn.functional.leaky_relu(f)

    return {'f': f}


# VECTOR o

  def compute_Uoh(self, edges):
    h = edges.src['h']
    Uoh = self.U_o(h)
    return {'Uoh': Uoh}

  def compute_o(self,nodes):
    x = nodes.data['proc_node']
    Uoh_sum = nodes.data['Uoh_sum']
    Wx = self.W_o(x)
    o = Wx + Uoh_sum
    o = th.nn.functional.leaky_relu(o)
    return {'o': o}


# VECTOR u

  def compute_Uuh(self, edges):
    h = edges.src['h']
    Uuh = self.U_u(h)
    return {'Uuh': Uuh}

  def compute_u(self,nodes):
    x = nodes.data['proc_node']
    Uuh_sum = nodes.data['Uuh_sum']
    Wx = self.W_u(x)
    u = Wx + Uuh_sum
    u = th.tanh(u)
    return {'u': u}


# VECTOR c

  def compute_fc(self, edges):
    f = edges.data['f']
    c = edges.src['c']
    fc = f * c
    return {'fc': fc}

  def compute_c(self, nodes):
    i = nodes.data['i']
    u = nodes.data['u']
    fc_sum = nodes.data['fc_sum']  # Assuming 'fc_sum' already exists or is initialized.
    c = i * u
    c = fc_sum + c
    return {'c': c}


# VECTOR h

  def compute_h(self,nodes):
    o = nodes.data['o']
    c = nodes.data['c']
    c = th.tanh(c)
    h = o * c
    return {'h': h}


  def forward(self, g):
      """
      Forward step

      Arguments:
          g: the graph

      Returns:
          n x 2 tensor (n number of nodes in the graph) containing the update
              for pressure (first column) and the update for the flowrate
              (second column)

      """

  # ENCODE
      g.apply_nodes(self.encode_nodes)
      g.apply_edges(self.encode_edges)

  # PROCESS

      # Vector i
      g.apply_edges(self.compute_Uih)
      g.update_all(fn.copy_e('Uih', 'm'),
                    fn.sum('m', 'Uih_sum')) 
      g.apply_nodes(self.compute_i)

      # Vector f
      g.apply_edges(self.compute_f)

      # Vector o
      g.apply_edges(self.compute_Uoh)
      g.update_all(fn.copy_e('Uoh', 'm'),
                    fn.sum('m', 'Uoh_sum'))
      g.apply_nodes(self.compute_o)

      # Vector u
      g.apply_edges(self.compute_Uuh)
      g.update_all(fn.copy_e('Uuh', 'm'),
                    fn.sum('m', 'Uuh_sum'))
      g.apply_nodes(self.compute_u)

      # Vector c
      g.apply_edges(self.compute_fc)
      g.update_all(fn.copy_e('fc', 'm'),
                    fn.sum('m', 'fc_sum'))
      g.apply_nodes(self.compute_c)

      # Vector h
      g.apply_nodes(self.compute_h)

  # DECODE
      g.apply_nodes(self.decode_nodes)

      return g.ndata['pred_labels']