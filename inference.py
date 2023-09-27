# ignore_header_test
# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from LSTM import GLSTMCell
from torch.cuda.amp import GradScaler
from generate_dataset import generate_normalized_graphs
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
import hydra
from omegaconf import DictConfig
import json
import time
import copy
from tqdm import tqdm


def evaluate_model(cfg, logger, model, params, graphs):
    rollout = Rollout(logger, cfg, model, params, graphs)
    testset = params["test_split"]
    ep_tot = 0
    eq_tot = 0
    for graph in tqdm(testset, desc="Testing graphs", colour="green"):
        rollout.predict(graph, do_print=False)
        rollout.denormalize()
        ep, eq = rollout.compute_errors(do_print=False)
        ep_tot += ep
        eq_tot += eq
    ep_tot = ep_tot/len(testset)
    eq_tot = eq_tot/len(testset)
    logger.info(f"Average relative error in pressure: {ep_tot * 100}%")
    logger.info(f"Average relative error in flowrate: {eq_tot * 100}%")
    return ep_tot, eq_tot
        

def denormalize(tensor, mean, stdv):
    """Denormalize a tensor given a mean and a standard deviation.
       denormalized_tensor = (tensor * stdv) + mean

    Arguments:
        tensor: tensor to denormalize
        mean: mean used for normalization
        stdv: standard deviation used for normalization

    Returns:
        denormalized tensor
    """
    return tensor * stdv + mean

def load_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate the model
    model = GLSTMCell(cfg)

    if cfg.performance.jit:
        model = torch.jit.script(model).to(device)

    else:
        model = model.to(device)

    scaler = GradScaler()
    # enable eval mode
    model.eval()

    # load checkpoint
    _ = load_checkpoint(
        os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
        models=model,
        device=device,
        scaler=scaler,
    )

    return model


class Rollout:
    def __init__(self, logger, cfg, model, params = None, graphs = None):
        """Performs the rollout phase on the geometry specified in
        'config.yaml' (testing.graph) and computes the error"""

        # set device

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger = logger
        if logger != None:
            logger.info(f"Using {self.device} device")

        if params == None:
            params = json.load(open("checkpoints/parameters.json"))
        
        if graphs == None:
            norm_type = {"features": "normal", "labels": "normal"}
            graphs, _ = generate_normalized_graphs(
                "raw_dataset/graphs/",
                norm_type,
                cfg.training.geometries,
                cfg,
                params["statistics"],
            )
        graph = graphs[list(graphs)[0]]

        self.model = model

        infeat_nodes = graph.ndata["nfeatures"].shape[1] + 1
        infeat_edges = graph.edata["efeatures"].shape[1]
        nout = 2
        nodes_features = [
            "area",
            "tangent",
            "type",
            "T",
            "dip",
            "sysp",
            "resistance1",
            "capacitance",
            "resistance2",
            "loading",
        ]

        edges_features = ["rel_position", "distance", "type"]

        params["infeat_nodes"] = infeat_nodes
        params["infeat_edges"] = infeat_edges
        params["out_size"] = nout
        params["node_features"] = nodes_features
        params["edges_features"] = edges_features
        params["rate_noise"] = 100
        params["rate_noise_features"] = 1e-5
        params["stride"] = 5

        self.graphs = graphs

        self.params = params
        self.var_identifier = {"p": 0, "q": 1}

    def compute_average_branches(self, graph, flowrate):
        """
        Average flowrate over branch nodes

        Arguments:
            graph: DGL graph
            flowrate: 1D tensor containing nodal flow rate values

        """
        branch_id = graph.ndata["branch_id"].cpu().detach().numpy()
        bmax = np.max(branch_id)
        for i in range(bmax + 1):
            idxs = np.where(branch_id == i)[0]
            rflowrate = torch.mean(flowrate[idxs])
            flowrate[idxs] = rflowrate

    def predict(self, graph_name, do_print = True):
        """
        Perform rollout phase for a single graph in the dataset

        Arguments:
            graph_name: the graph name.

        """
        graph = self.graphs[graph_name]
        graph = copy.deepcopy(graph.to(self.device))
        self.graph = graph

        ntimes = graph.ndata["pressure"].shape[-1]
        nnodes = graph.ndata["pressure"].shape[0]

        self.pred = torch.zeros((nnodes, 2, ntimes), device=self.device)
        self.exact = graph.ndata["nfeatures"][:, 0:2, :]
        self.pred[:, 0:2, 0] = graph.ndata["nfeatures"][:, 0:2, 0]

        inmask = graph.ndata["inlet_mask"].bool()
        invar = graph.ndata["nfeatures"][:, :, 0].clone().squeeze()
        efeatures = graph.edata["efeatures"].squeeze()
        graph.edata["efeatures"] = efeatures
        nnodes = inmask.shape[0]
        nf = torch.zeros((nnodes, 1), device=self.device)
        start = time.time()
        for i in range(ntimes - 1):
            # set loading variable (check original paper for reference)
            invar[:, -1] = graph.ndata["nfeatures"][:, -1, i]
            # we set the next flow rate at the inlet (boundary condition)
            nf[inmask, 0] = graph.ndata["nfeatures"][inmask, 1, i + 1]
            nfeatures = torch.cat((invar, nf), 1)
            graph.ndata["nfeatures_w_bcs"] = nfeatures
            pred = self.model(graph)
            invar[:, 0:2] += pred
            # we set the next flow rate at the inlet since that is known
            invar[inmask, 1] = graph.ndata["nfeatures"][inmask, 1, i + 1]
            # flow rate must be constant in branches
            self.compute_average_branches(graph, invar[:, 1])

            self.pred[:, :, i + 1] = invar[:, 0:2]

        end = time.time()
        if do_print:
            self.logger.info(f"Rollout took {end - start} seconds!")

    def denormalize(self):
        """
        Denormalize predicted and exact pressure and flow rate values. This
        function must be called after 'predict'.

        Arguments:
            graph_name: the graph name.

        """
        self.pred[:, 0, :] = denormalize(
            self.pred[:, 0, :],
            self.params["statistics"]["pressure"]["mean"],
            self.params["statistics"]["pressure"]["stdv"],
        )
        self.pred[:, 1, :] = denormalize(
            self.pred[:, 1, :],
            self.params["statistics"]["flowrate"]["mean"],
            self.params["statistics"]["flowrate"]["stdv"],
        )
        self.exact[:, 0, :] = denormalize(
            self.exact[:, 0, :],
            self.params["statistics"]["pressure"]["mean"],
            self.params["statistics"]["pressure"]["stdv"],
        )
        self.exact[:, 1, :] = denormalize(
            self.exact[:, 1, :],
            self.params["statistics"]["flowrate"]["mean"],
            self.params["statistics"]["flowrate"]["stdv"],
        )

    def compute_errors(self, do_print = True):
        """
        Compute errors in pressure and flow rate. This function must be called
        after 'predict' and 'denormalize'. The errors are computed as l2 errors
        at the branch nodes for all timesteps.

        """
        bm = torch.reshape(self.graph.ndata["branch_mask"], (-1, 1, 1))
        bm = bm.repeat(1, 2, self.pred.shape[2])
        diff = (self.pred - self.exact) * bm
        errs = torch.sum(torch.sum(diff**2, axis=0), axis=1)
        errs = errs / torch.sum(torch.sum((self.exact * bm) ** 2, axis=0), axis=1)
        errs = torch.sqrt(errs)

        if do_print:
            self.logger.info(f"Relative error in pressure: {errs[0] * 100}%")
            self.logger.info(f"Relative error in flowrate: {errs[1] * 100}%")

        return errs[0], errs[1]

    def plot(self, idx):
        """
        Creates plot of pressure and flow rate at the node specified with the
        idx parameter.

        Arguments:
            idx: Index of the node to plot pressure and flow rate at.

        """
        p_pred_values = []
        q_pred_values = []
        p_exact_values = []
        q_exact_values = []

        bm = self.graph.ndata["branch_mask"].bool()

        nsol = self.pred.shape[2]
        if torch.cuda.is_available():
            for isol in range(nsol):
                # if load[isol] == 0:
                p_pred_values.append(self.pred[bm, 0, isol][idx].cpu().detach().numpy())
                q_pred_values.append(self.pred[bm, 1, isol][idx].cpu().detach().numpy())
                p_exact_values.append(self.exact[bm, 0, isol][idx].cpu().detach().numpy())
                q_exact_values.append(self.exact[bm, 1, isol][idx].cpu().detach().numpy())
        else:
            for isol in range(nsol):
                # if load[isol] == 0:
                p_pred_values.append(self.pred[bm, 0, isol][idx])
                q_pred_values.append(self.pred[bm, 1, isol][idx])
                p_exact_values.append(self.exact[bm, 0, isol][idx])
                q_exact_values.append(self.exact[bm, 1, isol][idx])

        plt.figure()
        ax = plt.axes()
        ax.plot(p_pred_values, label="pred")
        ax.plot(p_exact_values, label="exact")
        ax.legend()
        plt.savefig("pressure.png", bbox_inches="tight")

        plt.figure()
        ax = plt.axes()
        ax.plot(q_pred_values, label="pred")
        ax.plot(q_exact_values, label="exact")
        ax.legend()
        plt.savefig("flowrate.png", bbox_inches="tight")

def do_rollout(cfg, logger, model):
    """
    Perform rollout phase.

    Arguments:
        cfg: Dictionary containing problem parameters.

    """
    rollout = Rollout(logger, cfg, model)
    rollout.predict(cfg.testing.graph)
    rollout.denormalize()
    rollout.compute_errors()
    rollout.plot(idx=5)
    return rollout

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Rollout started...")
    model = load_model(cfg)

    rollout = do_rollout(cfg, logger, model)
    evaluate_model(cfg, logger, model, rollout.params, rollout.graphs)


"""
The main function perform the rollout phase on the geometry specified in
'config.yaml' (testing.graph) and computes the error.
"""
if __name__ == "__main__":
    main()
