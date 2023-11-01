import os
import sys
import torch as th
from torch.nn.modules.module import Module
from torch.nn import LayerNorm
from torch.nn import Linear
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


class TRANSFORMERCell(Module):
    """
    Graph LSTM cell.

    This class computes pressure and flowrate updates given the previous system
    state.

    """

    def __init__(self, cfg):
        super(TRANSFORMERCell, self).__init__()

        self.encoder_nodes_reduction = MLP(
            cfg.architecture.in_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.decoder_nodes_reduction= MLP(
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_TRANSFORMER,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
            False,
        )
        self.encoder_edges_reduction = MLP(
            cfg.architecture.edge_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.encoder_edges_recovery = MLP(
            cfg.architecture.edge_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.decoder_nodes_recovery = MLP(
            cfg.architecture.latent_size_TRANSFORMER,
            cfg.architecture.out_size,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
            False,
        )
        self.encoder_nodes_recovery = MLP(
            cfg.architecture.latent_size_TRANSFORMER,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )

        self.processor_nodes_reduction = th.nn.ModuleList()
        self.processor_nodes_recovery = th.nn.ModuleList()
        self.processor_edges_reduction = th.nn.ModuleList()
        self.processor_edges_recovery = th.nn.ModuleList()
        self.process_iters = cfg.architecture.process_iterations
        for i in range(self.process_iters):
            def generate_proc_MLP(in_feat, out_feat):
                return MLP(in_feat,
                           out_feat,
                           cfg.architecture.latent_size_mlp,
                           cfg.architecture.number_hidden_layers_mlp)

            lsgnn = cfg.architecture.latent_size_gnn
            self.processor_nodes_reduction.append(generate_proc_MLP(lsgnn * 2,
                                        cfg.architecture.latent_size_gnn))
            self.processor_edges_reduction.append(generate_proc_MLP(lsgnn * 3,
                                        lsgnn))
            
            self.processor_nodes_recovery.append(generate_proc_MLP(lsgnn * 2,
                                        cfg.architecture.latent_size_gnn))
            self.processor_edges_recovery.append(generate_proc_MLP(lsgnn * 3,
                                        lsgnn))

        latent_gnn_dim = cfg.architecture.latent_size_gnn
        hidden_dim_l = cfg.architecture.hidden_dim
        hidden_dim_h = cfg.architecture.hidden_dim
        self.hidden_dim = hidden_dim_h

    
    def encode_nodes_reduction(self, nodes):
        """
        Encode graph nodes

        Arguments:
            edges: graph nodes

        Returns:
            dictionary (key: 'proc_nodes', value: encoded features)

        """
        features = nodes.data["current_state"]
        enc_features = self.encoder_nodes_reduction(features)
        return {"proc_node": enc_features}
    
    def encode_nodes_recovery(self, nodes):
        """
        Encode graph nodes

        Arguments:
            edges: graph nodes

        Returns:
            dictionary (key: 'proc_nodes', value: encoded features)

        """
        features = nodes.data["R"]
        enc_features = self.encoder_nodes_recovery(features)
        return {"proc_node": enc_features}

    def encode_edges_reduction(self, edges):
        """
        Encode graph edges

        Arguments:
            edges: graph edges

        Returns:
            dictionary (key: 'proc_edge', value: encoded features)

        """
        features = edges.data["efeatures"]
        enc_features = self.encoder_edges_reduction(features)
        return {"proc_edge": enc_features}
    
    def encode_edges_recovery(self, edges):
        """
        Encode graph edges

        Arguments:
            edges: graph edges

        Returns:
            dictionary (key: 'proc_edge', value: encoded features)

        """
        features = edges.data["efeatures"]
        enc_features = self.encoder_edges_recovery(features)
        return {"proc_edge": enc_features}

    def decode_nodes_reduction(self, nodes):
        """
        Decode graph nodes

        Arguments:
            nodes: graph nodes

        Returns:
            dictionary (key: 'pred_labels', value: decoded features)

        """
        h = self.decoder_nodes_reduction(nodes.data["proc_node"])
        return {"h": h}
    
    
    def decode_nodes_recovery(self, nodes):
        """
        Decode graph nodes

        Arguments:
            nodes: graph nodes

        Returns:
            dictionary (key: 'pred_labels', value: decoded features)

        """
        h = self.decoder_nodes_recovery(nodes.data["h"])
        return {"h": h}
    
    
    def process_edges_reduction(self, edges, index):
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

        proc_edge = self.processor_edges_reduction[index](th.cat((f1, f2, f3), 1))
        # print(f1.shape)
        # print(proc_edge.shape)
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge': proc_edge}
    
    
    def process_edges_recovery(self, edges, index):
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

        proc_edge = self.processor_edges_recovery[index](th.cat((f1, f2, f3), 1))
        # add residual connection
        proc_edge = proc_edge + f1
        return {'proc_edge': proc_edge}

    def process_nodes_reduction(self, nodes, index):
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
        proc_node = self.processor_nodes_reduction[index](th.cat((f1, f2), 1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node': proc_node}

    def process_nodes_recovery(self, nodes, index):
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
        proc_node = self.processor_nodes_recovery[index](th.cat((f1, f2), 1))
        # add residual connection
        proc_node = proc_node + f1
        return {'proc_node': proc_node}

    def graph_reduction(self, g):
        """
        Forward step
        self.encoder_edges_reduction = MLP(
            cfg.architecture.edge_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.encoder_edges_recovery = MLP(
            cfg.architecture.edge_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        Arguments:
            g: the graph

        Returns:
            n x 2 tensor (n number of nodes in the graph) containing the update
                for pressure (first column) and the update for the flowrate
                (second column)

        """
        # ENCODE
        g.apply_nodes(self.encode_nodes_reduction)
        g.apply_edges(self.encode_edges_reduction)


        #PROCESS   
   
        for index in range(self.process_iters):
            def process_edges_reduction(edges):
                return self.process_edges_reduction(edges, index)
            def process_nodes_reduction(nodes):
                return self.process_nodes_reduction(nodes, index)
            # compute junction-branch interactions
            g.apply_edges(process_edges_reduction)
            g.update_all(fn.copy_e('proc_edge', 'm'), 
                         fn.sum('m', 'pe_sum'))
            g.apply_nodes(process_nodes_reduction)


        # DECODE
        g.apply_nodes(self.decode_nodes_reduction)
        
        z = th.reshape(g.ndata["h"][g.ndata["pivotal_nodes"].bool(),:],(-1,))
        return z
    
    
    def graph_recovery(self, g, z):

        # controllare se g e' grafo singolo o batch
        graphs = dgl.unbatch(g)

        # interpolation
        npnodes = th.sum(g.ndata["pivotal_nodes"])
        # print(z.shape)
        # print(npnodes)
        H = th.reshape(z,(npnodes,-1))

        R = th.zeros((g.ndata["pivotal_weights"].shape[0], 
                      H.shape[1]))

        offset_h = 0
        offset_w = 0
        for single_graph in graphs:
            W = single_graph.ndata["pivotal_weights"]
            npnodes_per_graph = W.shape[1]
            single_H = H[offset_h:offset_h + npnodes_per_graph,:]
            w_norm = th.sum(W,axis=1).unsqueeze(axis=1)
            R[offset_w:offset_w + W.shape[0]] = th.div(th.matmul(W,single_H), w_norm)

            offset_w += W.shape[0]
            offset_h += npnodes_per_graph

        # W = g.ndata["pivotal_weights"] 


        # w_norm = th.sum(W,axis=1).unsqueeze(axis=1)
        # # print(th.matmul(W,H).shape)
        # # print(w_norm.shape)
        # R = th.div(th.matmul(W,H), w_norm)
        # # R = th.matmul(W,H)/w_norm

        g.ndata["R"] = R

        # ENCODE
        g.apply_nodes(self.encode_nodes_recovery)
        g.apply_edges(self.encode_edges_recovery)


        #PROCESS   
   
        for index in range(self.process_iters):
            def process_edges_recovery(edges):
                return self.process_edges_recovery(edges, index)
            def process_nodes_recovery(nodes):
                return self.process_nodes_recovery(nodes, index)
            # compute junction-branch interactions
            g.apply_edges(process_edges_recovery)
            g.update_all(fn.copy_e('proc_edge', 'm'), 
                         fn.sum('m', 'pe_sum'))
            g.apply_nodes(process_nodes_recovery)


        # DECODE
        g.apply_nodes(self.decode_nodes_recovery)
        return g.ndata["h"]
    
    def forward(self, g):
        encoded = self.graph_reduction(g)
        decoded = self.graph_recovery(g, encoded)
        return decoded
