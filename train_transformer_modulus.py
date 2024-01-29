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
import dgl
import torch
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import random
import time, os
import numpy as np
import hydra
from inference import evaluate_model

from modulus.distributed.manager import DistributedManager

#  from modulus.models.meshgraphnet import MeshGraphNet
from transformer import TransformerCell

# from modulus.datapipes.gnn.mgn_dataset import MGNDataset
import generate_dataset as gd
from generate_dataset import generate_normalized_graphs
from generate_dataset import train_test_split
from generate_dataset import Bloodflow1DDataset

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
import json
from omegaconf import DictConfig
import uuid
import os
from torch.autograd import Function
from AE import AECell

def mse(input, target, mask=1):
    """
    Mean square error.

    This is defined as the ((input - target)**2).mean()

    Arguments:
        input: first tensor
        target: second tensor (ideally, the result we are trying to match)
        mask: tensor of weights for loss entries with same size as input and
              target.

    Returns:
        The mean square error

    """
    return (mask * (input - target) ** 2).mean()

def dgl_collate(batch):
    graphs = [item for item in batch]
    return dgl.batch(graphs)
        
class MGNTrainer:
    def __init__(self, logger, cfg, dist):
        # set device
        self.device = dist.device
        logger.info(f"Using {self.device} device")

        norm_type = {"features": "normal", "labels": "normal"}

        graphs, params = generate_normalized_graphs(
            cfg.work_directory + "/raw_dataset/graphs/", norm_type, cfg.training.geometries, cfg
        )

        self.graphs = graphs
        graph = graphs[list(graphs)[0]]

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

        trainset, testset = train_test_split(graphs, cfg.training.train_test_split)
        params["train_split"] = trainset
        params["test_split"] = testset

        self.train_graphs = [graphs[gname] for gname in trainset]
        self.test_graphs = [graphs[gname] for gname in testset]

        traindataset = Bloodflow1DDataset(self.train_graphs, params, trainset)

        self.stride = traindataset.mint

        # instantiate dataloader
        self.train_dataloader = DataLoader(
            traindataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dgl_collate
        )


        # instantiate the model
        self.model = TransformerCell(cfg)

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.transformer_architecture.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.transformer_architecture.epochs,
            eta_min=cfg.transformer_architecture.lr * cfg.transformer_architecture.lr_decay,
        )
        self.scaler = GradScaler()

        # load checkpoint
        self.epoch_init = load_checkpoint(
            os.path.join(cfg.transformer_architecture.checkpoints_ckpt_path, cfg.transformer_architecture.checkpoints_ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

        self.params = params
        self.cfg = cfg


    def backward(self, loss):
        """
        Perform backward pass.

        Arguments:
            loss: loss value.

        """
        # backward pass
        if self.cfg.performance.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping (for stabilization and performance).
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0, norm_type = 2.0, error_if_nonfinite = True)
            self.optimizer.step()


    def train(self, mu, z_0, Z): 
        """
        Perform one training iteration over one graph. The training is performed
        over multiple timesteps, where the number of timesteps is specified in
        the 'stride' parameter.

        Arguments:
            graph: the desired graph.

        Returns:
            loss: loss value.

        """

        # Incremental loss function
        if self.cfg.transformer_architecture.incremental_loss:
            for idx_t in range(2, self.cfg.transformer_architecture.N_timesteps):
                loss = self.cfg.transformer_architecture.threshold + 1.
                while (loss > self.cfg.transformer_architecture.threshold):
                    self.optimizer.zero_grad()
                    pred = self.model(mu, z_0, idx_t)
                    loss = mse(pred[:, 2:, :], Z[:, 1:idx_t, :], mask = 1)
                    self.backward(loss)
        else:
            self.optimizer.zero_grad()
            pred = self.model(mu, z_0)
            loss = mse(pred[:, 2:, :], Z[:, 1:, :], mask = 1)
            self.backward(loss)

        return loss


@hydra.main(version_base=None, config_path=".", config_name="config")
def read_cfg(cfg: DictConfig):
    return cfg

def do_training(cfg, dist):
    """
    Perform training over all graphs in the dataset.

    Arguments:
        cfg: Dictionary of parameters.

    """
    # initialize loggers
    logger = PythonLogger("main")
    logger.file_logging()

    # initialize trainer
    trainer = MGNTrainer(logger, cfg, dist)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate the model
    AE_model = AECell(cfg)

    if cfg.performance.jit:
        AE_model = torch.jit.script(AE_model).to(device)
    else:
        AE_model = AE_model.to(device)

    scaler = GradScaler()
    # enable eval mode
    AE_model.eval()

    if cfg.hyperparameter_optimization.flag == "True":

        #vecchia_directory = os.getcwd()
        #nuova_directory = "/expanse/lustre/scratch/aiacovelli/temp_project/tdm_grom/tdm-gROM"
        # os.chdir("..")
        #os.chdir(nuova_directory)

        # load checkpoint
        load_epoch = load_checkpoint(
            "/expanse/lustre/scratch/aiacovelli/temp_project/tdm_grom/tdm-gROM/checkpoints/model.pt/",
            models=AE_model,
            device=device,
            scaler=scaler,
        )
        '''
        load_epoch = load_checkpoint(
            os.path.join(nuova_directory, cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=AE_model,
            device=device,
            scaler=scaler,
        )
        '''

        #directory = os.getcwd() 
        #print("nuova dir", directory)

        #os.chdir(vecchia_directory)
        #print("vecchia dir", vecchia_directory)


        if load_epoch == 0:
            raise ValueError("Checkpoints not found!")

            
    else:
        # load checkpoint
        load_epoch = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=AE_model,
            device=device,
            scaler=scaler,
        )

        directory = os.getcwd() 
        print(directory) # /expanse/lustre/scratch/aiacovelli/temp_project/tdm_grom/tdm-gROM

        if load_epoch == 0:
            raise ValueError("Checkpoints not found!")

    # allocate variables
    npnodes = torch.sum(trainer.train_graphs[0].ndata["pivotal_nodes"]).item()
    ngraphs_train = len(trainer.train_graphs)
    total_nodes_train = npnodes * ngraphs_train
    Z = torch.zeros(total_nodes_train, cfg.transformer_architecture.N_timesteps, cfg.architecture.latent_size_AE, device=trainer.device)
    z_0 = torch.zeros(total_nodes_train, cfg.architecture.latent_size_AE, device=trainer.device)
    mu = torch.zeros(total_nodes_train, cfg.transformer_architecture.N_timesteps, device=trainer.device)
    for idx_g in range(ngraphs_train):
        # read graph
        graph = trainer.train_graphs[idx_g]
        graph = graph.to(device)

        states = [graph.ndata["nfeatures"][:, :, 0]]
        graph.edata["efeatures"] = graph.edata["efeatures"].squeeze()

        for istride in range(trainer.stride):
            # inference on encoder
            ns = graph.ndata["nfeatures"][:, :, istride]
            graph.ndata["current_state"] = ns

            with torch.no_grad():
                reduction_output = AE_model.graph_reduction(graph)
                # print(f"Reduction output size: {reduction_output.size()}")  
                # print(f"Before reshaping Z: {Z.size()}")  
                # print(f"Target slice size: {Z[idx_g * npnodes : (idx_g + 1) * npnodes, istride, :].size()}")
                # print(f"Before reshaping npnodes: {npnodes}")
                Z[idx_g * npnodes : (idx_g + 1) * npnodes, istride, :] = torch.reshape(AE_model.graph_reduction(graph), (npnodes, cfg.architecture.latent_size_AE))

        # impose boundary condition
        mu[idx_g * npnodes : (idx_g + 1) * npnodes, :] = graph.ndata["nfeatures"][0, 1, :]

        # impose initial condition
        z_0[idx_g * npnodes : (idx_g + 1) * npnodes, :] = Z[idx_g * npnodes : (idx_g + 1) * npnodes, 0, :]

    # training loop
    start = time.time()
    logger.info("Training started...")
    loss_vector = []  # Initialize an empty list to store loss values
    for epoch in range(trainer.epoch_init, cfg.transformer_architecture.epochs):
        # Loop over batches
        #for idx_g in random.sample(range(ngraphs_train), ngraphs_train):
        #   loss = trainer.train(mu[idx_g * npnodes : (idx_g + 1) * npnodes, :], z_0[idx_g * npnodes : (idx_g + 1) * npnodes, :], Z[idx_g * npnodes : (idx_g + 1) * npnodes, :, :])
        loss = trainer.train(mu, z_0, Z)
        loss_vector.append(loss.cpu().detach().numpy())

        if torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated()
        else:
            max_memory_allocated = psutil.virtual_memory()[3]

        logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}, memory allocated: {max_memory_allocated/1024**3:.2f} GB"
        )

        if cfg.training.output_interval != -1:
            if (epoch % cfg.training.output_interval) == 0 or epoch == 0 or epoch == (cfg.transformer_architecture.epochs-1): 
                # save checkpoint
                save_checkpoint(
                    os.path.join(cfg.transformer_architecture.checkpoints_ckpt_path, cfg.transformer_architecture.checkpoints_ckpt_name),
                    models=trainer.model,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler,
                    scaler=trainer.scaler,
                    epoch=epoch,
                )
        start = time.time()
        trainer.scheduler.step()

        def default(obj):
            if isinstance(obj, torch.Tensor):
                return default(obj.detach().numpy())
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            print(obj)
            return TypeError("Token is not serializable")

        if cfg.training.output_interval != -1:
            with open(cfg.transformer_architecture.checkpoints_ckpt_path + "/parameters.json", "w") as outf:
                json.dump(trainer.params, outf, default=default, indent=4)
    logger.info("Training completed!")

    # Plot loss_vector
    plt.figure()
    ax = plt.axes()
    ax.semilogy(loss_vector, label="loss")
    ax.legend()
    plt.savefig("loss.png", bbox_inches="tight")

    # evaluate model
    ep = 0
    eq = 0
    ngraphs_test = len(trainer.test_graphs)
    for idx_g in range(ngraphs_test):
        # read graph
        graph = trainer.test_graphs[idx_g]
        graph = graph.to(device)

        states = [graph.ndata["nfeatures"][:, :, 0]]
        graph.edata["efeatures"] = graph.edata["efeatures"].squeeze()
        
        # numerical simulation in time with transformer
        ns = graph.ndata["nfeatures"][:, :, 0]
        graph.ndata["current_state"] = ns
        with torch.no_grad():
            mu = graph.ndata["nfeatures"][0, 1, :].unsqueeze(0).repeat(npnodes, 1)
            z_0 = torch.reshape(AE_model.graph_reduction(graph), (npnodes, cfg.architecture.latent_size_AE))
            pred = trainer.model(mu, z_0)[:, 1:, :]
        
        # inference on decoder
        decoded = torch.zeros(graph.number_of_nodes(), cfg.architecture.out_size, cfg.transformer_architecture.N_timesteps, device=trainer.device)
        for istride in range(trainer.stride):
            ns = graph.ndata["nfeatures"][:, :, istride]
            graph.ndata["current_state"] = ns
            ## 
            _ = torch.reshape(AE_model.graph_reduction(graph), (npnodes, cfg.architecture.latent_size_AE))
            ##
            with torch.no_grad():
                decoded[:, :, istride] = AE_model.graph_recovery(graph, torch.reshape(pred[:, istride, :], (-1,)))
                #print(decoded[:,:,istride])
        decoded[:, 0, :] = decoded[:, 0, :] * trainer.params["statistics"]["pressure"]["stdv"] + trainer.params["statistics"]["pressure"]["mean"]
        decoded[:, 1, :] = decoded[:, 1, :] * trainer.params["statistics"]["flowrate"]["stdv"] + trainer.params["statistics"]["flowrate"]["mean"]
        graph.ndata["nfeatures"][:, 0, :] = graph.ndata["nfeatures"][:, 0, :] * trainer.params["statistics"]["pressure"]["stdv"] + trainer.params["statistics"]["pressure"]["mean"]        
        graph.ndata["nfeatures"][:, 1, :] = graph.ndata["nfeatures"][:, 1, :] * trainer.params["statistics"]["flowrate"]["stdv"] + trainer.params["statistics"]["flowrate"]["mean"]
        diff = decoded - graph.ndata["nfeatures"][:, 0:2, :]
        errs = torch.sum(torch.sum(diff ** 2, axis=0), axis=1)
        errs = errs / torch.sum(torch.sum(graph.ndata["nfeatures"][:, 0:2, :] ** 2, axis=0), axis=1)
        errs = torch.sqrt(errs)
        ep += errs[0] 
        eq += errs[1]

    ep = ep / ngraphs_test
    eq = eq / ngraphs_test

    print((ep + eq) / 2)
    
    # PLOT
    p_pred_values = []
    q_pred_values = []
    p_exact_values = []
    q_exact_values = []

    pred = decoded
    exact = graph.ndata["nfeatures"][:, 0:2, :]
    idx = 5

    nsol = pred.shape[2]
    if torch.cuda.is_available():
        for isol in range(nsol):
            # if load[isol] == 0:
            p_pred_values.append(pred[:, 0, isol][idx].cpu().detach().numpy())
            q_pred_values.append(pred[:, 1, isol][idx].cpu().detach().numpy())
            p_exact_values.append(exact[:, 0, isol][idx].cpu().detach().numpy())
            q_exact_values.append(exact[:, 1, isol][idx].cpu().detach().numpy())
    else:
        for isol in range(nsol):
            # if load[isol] == 0:
            p_pred_values.append(pred[:, 0, isol][idx])
            q_pred_values.append(pred[:, 1, isol][idx])
            p_exact_values.append(exact[:, 0, isol][idx])
            q_exact_values.append(exact[:, 1, isol][idx])

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

    return (ep + eq) / 2


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    do_training(cfg, dist)

if __name__ == "__main__":
    main()