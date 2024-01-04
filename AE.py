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
import random


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
        seed = 1
        th.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
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
        # print("After input layer:", f)
        f = F.leaky_relu(f)
        # print("After leaky_relu:", f)

        for i in range(self.n_h_layers):
            f = self.hidden_layers[i](f)
            f = F.leaky_relu(f)

        # enc_features = self.dropout(enc_features)
        f = self.output(f)

        if self.normalize:
            f = self.norm(f)

        return f


class AECell(Module):
    """
    Graph LSTM cell.

    This class computes pressure and flowrate updates given the previous system
    state.

    """

    def __init__(self, cfg):
        super(AECell, self).__init__()

        self.encoder_nodes_reduction = MLP(
            cfg.architecture.in_feats,
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
        )
        self.decoder_nodes_reduction= MLP(
            cfg.architecture.latent_size_gnn,
            cfg.architecture.latent_size_AE,
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
            cfg.architecture.latent_size_AE, # Cancella AE e ripristina: gnn
            cfg.architecture.out_size,
            cfg.architecture.latent_size_mlp,
            cfg.architecture.number_hidden_layers_mlp,
            False,
        )
        self.encoder_nodes_recovery = MLP(
            cfg.architecture.latent_size_AE,
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
        # print("ENC NODES RED: Pre MLP", features)
        enc_features = self.encoder_nodes_reduction(features)
        # print("ENC NODES RED: Post MLP", enc_features)
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
        # print("ENC EDGES REC: Pre MLP", features)
        enc_features = self.encoder_nodes_recovery(features)
        # print("ENC EDGES REC: Post MLP", enc_features)
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
        # print("ENC EDGES RED: Pre MLP", features)
        enc_features = self.encoder_edges_reduction(features)
        # print("ENC EDGES RED: Post MLP", enc_features)
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
        h = self.decoder_nodes_recovery(nodes.data["h"]) #ripristina: h = self.decoder_nodes_recovery(nodes.data["proc_node"])
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

        #print("NODES: Pre MLP f1", f1)
        #print("NODES: Pre MLP f2", f2)

        proc_node = self.processor_nodes_reduction[index](th.cat((f1, f2), 1))
        # add residual connection
        proc_node = proc_node + f1

        #print("NODES: Post MLP", proc_node)

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

        print("h", g.ndata["h"] )

        # controllare se g e' grafo singolo o batch
        graphs = dgl.unbatch(g)

        # interpolation
        npnodes = th.sum(g.ndata["pivotal_nodes"])
        # print(z.shape)
        # print(npnodes)
        H = th.reshape(z,(npnodes,-1))


        tensor = g.ndata["pivotal_weights"]
        elements_inf = th.isinf(tensor)
        # Sostituisci gli elementi "inf" con il massimo per 100
        tensor[elements_inf] = 150.4765*100
        g.ndata["pivotal_weights"] = tensor
        #sostituisci con valori H i valori nan di R, nel ciclo for
        #indices = []
        #for index in range(g.ndata["pivotal_nodes"].shape[0]):
        #    if (g.ndata["pivotal_nodes"][index] == 1):
        #        indices.append(index)
        #print(indices)
        #print(len(indices))

        R = th.zeros((g.ndata["pivotal_weights"].shape[0], 
                      H.shape[1]), device=z.device)

        offset_h = 0
        offset_w = 0
        # print("Pre R: ", R)
        for single_graph in graphs:
            W = single_graph.ndata["pivotal_weights"]
            npnodes_per_graph = W.shape[1]
            single_H = H[offset_h:offset_h + npnodes_per_graph,:]
            # print("W: ", W)
            w_norm = th.sum(W,axis=1).unsqueeze(axis=1)
            R[offset_w:offset_w + W.shape[0]] = th.div(th.matmul(W,single_H), w_norm)
            #print("R pre: ", R[indices, :])
            #R[indices, :] = H
            #print("R post: ", R)
            #print("NAN: ", th.nonzero(th.isnan(R)))
            offset_w += W.shape[0]
            offset_h += npnodes_per_graph
        # print("Post R: ", R)

        g.ndata["R"] = R
        print("R", R)

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
