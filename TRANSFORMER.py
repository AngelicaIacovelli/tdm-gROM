import torch as th
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import dgl.function as fn




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


# class _GLSTMCell(Module):
#     """
#     Graph LSTM cell.

#     This class computes pressure and flowrate updates given the previous system
#     state.

#     """

#     def __init__(self, cfg):
#         super(GLSTMCell, self).__init__()

#         self.encoder_nodes = MLP(
#             cfg.architecture.in_feats,
#             cfg.architecture.latent_size_gnn,
#             cfg.architecture.latent_size_mlp,
#             cfg.architecture.number_hidden_layers_mlp,
#         )
#         self.encoder_edges = MLP(
#             cfg.architecture.edge_feats,
#             cfg.architecture.latent_size_gnn,
#             cfg.architecture.latent_size_mlp,
#             cfg.architecture.number_hidden_layers_mlp,
#         )
#         self.output = MLP(
#             cfg.architecture.hidden_dim,
#             cfg.architecture.out_size,
#             cfg.architecture.latent_size_mlp,
#             cfg.architecture.number_hidden_layers_mlp,
#             False,
#         )

#         latent_gnn_dim = cfg.architecture.latent_size_gnn
#         hidden_dim_l = cfg.architecture.hidden_dim
#         hidden_dim_h = cfg.architecture.hidden_dim

#         self.W_i = Linear(latent_gnn_dim, hidden_dim_l, bias=True).float()
#         self.U_i = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

#         self.W_f = Linear(cfg.architecture.edge_feats, hidden_dim_l, bias=True).float()
#         self.U_f = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

#         self.W_o = Linear(latent_gnn_dim, hidden_dim_l, bias=True).float()
#         self.U_o = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

#         self.W_u = Linear(latent_gnn_dim, hidden_dim_l, bias=True).float()
#         self.U_u = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

#         self.autoloop_iterations = cfg.architecture.autoloop_iterations

#     def encode_nodes(self, nodes):
#         """
#         Encode graph nodes

#         Arguments:
#             edges: graph nodes

#         Returns:
#             dictionary (key: 'proc_nodes', value: encoded features)

#         """
#         features = nodes.data["nfeatures_w_bcs"]
#         enc_features = self.encoder_nodes(features)
#         return {"proc_node": enc_features}

#     def encode_edges(self, edges):
#         """
#         Encode graph edges

#         Arguments:
#             edges: graph edges

#         Returns:
#             dictionary (key: 'proc_edge', value: encoded features)

#         """
#         enc_features = self.encoder_edges(edges.data["efeatures"])
#         return {"proc_edge": enc_features}

#     def decode_nodes(self, nodes):
#         """
#         Decode graph nodes

#         Arguments:
#             nodes: graph nodes

#         Returns:
#             dictionary (key: 'pred_labels', value: decoded features)

#         """
#         h = self.output(nodes.data["h"])
#         return {"pred_labels": h}

#     # VECTOR i

#     def compute_Uih(self, edges):
#         weight_matrix_U = self.U_i.weight
#         # print("This is U_i: ", weight_matrix_U)

#         h = edges.src["h"]
#         Uih = self.U_i(h)
#         # print("This is Ui*h: ", Uih)
#         return {"Uih": Uih}

#     def compute_i(self, nodes):
#         x = nodes.data[
#             "proc_node"
#         ]  # dictionary (key: 'proc_node', value: processed features)

#         Uih_sum = nodes.data["Uih_sum"]
#         # print("This is Uih_sum, the sum of the neighbors' Uih per each node: ", Uih_sum)

#         weight_matrix_W = self.W_i.weight
#         bias = self.W_i.bias
#         # print("This is W_i: ", weight_matrix_W)
#         # print("This is the bias of W: ",bias)

#         Wx = self.W_i(x)
#         # print("This is Wx: ", Wx)

#         i = Wx + Uih_sum
#         # print("This is i: ", i)

#         i = th.sigmoid(i)
#         # print('i: ', i)
#         return {"i": i}

#     # VECTOR f

#     def compute_f(self, edges):
#         x = edges.data["efeatures"]
#         x = x.float()
#         Wx = self.W_f(x)

#         h = edges.src["h"]
#         Ufh = self.U_f(h)

#         f = Wx + Ufh
#         f = th.sigmoid(f)
#         # print("f: ",f)
#         return {"f": f}

#     # VECTOR o

#     def compute_Uoh(self, edges):
#         h = edges.src["h"]
#         Uoh = self.U_o(h)
#         return {"Uoh": Uoh}

#     def compute_o(self, nodes):
#         x = nodes.data["proc_node"]
#         Uoh_sum = nodes.data["Uoh_sum"]
#         Wx = self.W_o(x)
#         o = Wx + Uoh_sum
#         o = th.sigmoid(o)
#         # print("o: ", o)
#         return {"o": o}

#     # VECTOR u

#     def compute_Uuh(self, edges):
#         h = edges.src["h"]
#         Uuh = self.U_u(h)
#         return {"Uuh": Uuh}

#     def compute_u(self, nodes):
#         x = nodes.data["proc_node"]
#         Uuh_sum = nodes.data["Uuh_sum"]
#         Wx = self.W_u(x)
#         u = Wx + Uuh_sum
#         u = th.tanh(u)
#         # print("u: ", u)
#         return {"u": u}

#     # VECTOR c

#     def compute_fc(self, edges):
#         f = edges.data["f"]
#         c = edges.src["c"]
#         fc = f * (th.sigmoid(c))
#         return {"fc": fc}

#     def compute_c(self, nodes):
#         i = nodes.data["i"]
#         u = nodes.data["u"]
#         fc_sum = nodes.data[
#             "fc_sum"
#         ]  # Assuming 'fc_sum' already exists or is initialized.
#         c = i * u
#         c = fc_sum + c
#         # print("c: ", c)
#         return {"c": c}

#     # VECTOR h

#     def compute_h(self, nodes):
#         o = nodes.data["o"]
#         c = nodes.data["c"]
#         c = th.tanh(c)
#         h = o * c
#         # print("h: ", h)
#         return {"h": h}

#     def forward(self, g):
#         """
#         Forward step

#         Arguments:
#             g: the graph

#         Returns:
#             n x 2 tensor (n number of nodes in the graph) containing the update
#                 for pressure (first column) and the update for the flowrate
#                 (second column)

#         """

#         # ENCODE
#         g.apply_nodes(self.encode_nodes)
#         # g.apply_edges(self.encode_edges)

#         # LSTM Cell

#         for i in range(self.autoloop_iterations):

#             # Vector i
#             g.apply_edges(self.compute_Uih)
#             g.update_all(fn.copy_e("Uih", "m"), fn.sum("m", "Uih_sum"))
#             g.apply_nodes(self.compute_i)

#             # Vector f
#             g.apply_edges(self.compute_f)

#             # Vector o
#             g.apply_edges(self.compute_Uoh)
#             g.update_all(fn.copy_e("Uoh", "m"), fn.sum("m", "Uoh_sum"))
#             g.apply_nodes(self.compute_o)

#             # Vector u
#             g.apply_edges(self.compute_Uuh)
#             g.update_all(fn.copy_e("Uuh", "m"), fn.sum("m", "Uuh_sum"))
#             g.apply_nodes(self.compute_u)

#             # Vector c
#             g.apply_edges(self.compute_fc)
#             g.update_all(fn.copy_e("fc", "m"), fn.sum("m", "fc_sum"))
#             g.apply_nodes(self.compute_c)

#             # Vector h
#             g.apply_nodes(self.compute_h)

#         # DECODE
#         g.apply_nodes(self.decode_nodes)

#         return g.ndata["pred_labels"]



class GLSTMCell(Module):
    """
    Graph LSTM cell.

    This class computes pressure and flowrate updates given the previous system
    state.

    """

    def __init__(self, cfg):
        super(GLSTMCell, self).__init__()

        self.encoder_nodes = MLP(
            cfg.architecture.in_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.encoder_edges = MLP(
            cfg.architecture.edge_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.output = MLP(
            cfg.architecture.hidden_dim,
            cfg.architecture.out_size,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
            False,
        )

        self.processor_nodes = th.nn.ModuleList()
        self.processor_edges = th.nn.ModuleList()
        self.process_iters = cfg.architecture.process_iterations
        for i in range(self.process_iters):
            def generate_proc_MLP(in_feat, out_feat):
                return MLP(in_feat,
                           out_feat,
                           cfg.architecture.latent_size_mlp,
                           cfg.architecture.number_hidden_layers_mlp)

            lsgnn = cfg.architecture.latent_size_gnn
            self.processor_nodes.append(generate_proc_MLP(lsgnn + cfg.architecture.edge_feats,
                                        cfg.architecture.latent_size_gnn))
            self.processor_edges.append(generate_proc_MLP(lsgnn * 2 + cfg.architecture.edge_feats,
                                        cfg.architecture.edge_feats))

        latent_gnn_dim = cfg.architecture.latent_size_gnn
        hidden_dim_l = cfg.architecture.hidden_dim
        hidden_dim_h = cfg.architecture.hidden_dim
        self.hidden_dim = hidden_dim_h

        self.W_i = Linear(latent_gnn_dim, hidden_dim_l, bias=True).float()
        self.U_i = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

        self.W_f = Linear(cfg.architecture.edge_feats, hidden_dim_l, bias=True).float()
        self.U_f = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

        self.W_o = Linear(latent_gnn_dim, hidden_dim_l, bias=True).float()
        self.U_o = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

        self.W_u = Linear(latent_gnn_dim, hidden_dim_l, bias=True).float()
        self.U_u = Linear(hidden_dim_h, hidden_dim_l, bias=False).float()

        self.autoloop_iterations = cfg.architecture.autoloop_iterations

    def encode_nodes(self, nodes):
        """
        Encode graph nodes

        Arguments:
            edges: graph nodes

        Returns:
            dictionary (key: 'proc_nodes', value: encoded features)

        """
        features = nodes.data["nfeatures_w_bcs"]
        enc_features = self.encoder_nodes(features)
        return {"proc_node": enc_features}

    def encode_edges(self, edges):
        """
        Encode graph edges

        Arguments:
            edges: graph edges

        Returns:
            dictionary (key: 'proc_edge', value: encoded features)

        """
        features = edges.data["efeatures"]
        enc_features = self.encoder_edges(features)
        return {"proc_edge": enc_features}

    def decode_nodes(self, nodes):
        """
        Decode graph nodes

        Arguments:
            nodes: graph nodes

        Returns:
            dictionary (key: 'pred_labels', value: decoded features)

        """
        h = self.output(nodes.data["h"])
        return {"h": h}
    
    def process_edges(self, edges, index):
        """
        Process graph edges

        Arguments:
            edges: graph edges
            index: iteration index

        Returns:
            dictionary (key: 'proc_edge', value: processed features)

        """
        f1 = edges.data['proc_edge']
        f2 = edges.src['proc_node']
        f3 = edges.dst['proc_node']

        proc_edge = self.processor_edges[index](th.cat((f1, f2, f3), 1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge': proc_edge}

    def process_nodes(self, nodes, index):
        """
        Process graph nodes

        Arguments:
            nodes: graph nodes
            index: iteration index

        Returns:
            dictionary (key: 'proc_node', value: processed features)

        """
        f1 = nodes.data['proc_node']
        f2 = nodes.data['pe_sum']
        proc_node = self.processor_nodes[index](th.cat((f1, f2), 1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node': proc_node}
        
    def graph_reduction(self, g):
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


        #PROCESS   
   
        for index in range(self.process_iters):
            def process_edges(edges):
                return self.process_edges(edges, index)
            def process_nodes(nodes):
                return self.process_nodes(nodes, index)
            # compute junction-branch interactions
            g.apply_edges(process_edges)
            g.update_all(fn.copy_e('proc_edge', 'm'), 
                         fn.sum('m', 'pe_sum'))
            g.apply_nodes(process_nodes)


        # DECODE
        g.apply_nodes(self.decode_nodes)
        
        z = th.reshape(g.ndata["h"][g.ndata["pivotal_nodes"]],(-1,))
        return z
    
    
    def graph_recovery(self, g, z):


