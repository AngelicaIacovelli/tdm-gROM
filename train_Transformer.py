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
import psutil
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler
import time, os
import numpy as np
import hydra
from inference import evaluate_model

from modulus.distributed.manager import DistributedManager

#  from modulus.models.meshgraphnet import MeshGraphNet
from TRANSFORMER import TRANSFORMERCell

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

class SelfDeletingTempFile():
    def __init__(self):
        self.name = f"gradients/temp_file_{uuid.uuid4()}"
    
    def __del__(self):
        os.remove(self.name)

SAVE_ON_DISK_THRESHOLD = 50000000000
def pack_hook(tensor):
    if tensor.numel() < SAVE_ON_DISK_THRESHOLD:
        return tensor
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(temp_file):
    if isinstance(temp_file, torch.Tensor):
        return temp_file
    return torch.load(temp_file.name)

def mse(input, target, mask):
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

        train_graphs = [graphs[gname] for gname in trainset]
        traindataset = Bloodflow1DDataset(train_graphs, params, trainset)

        self.stride = traindataset.mint

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            traindataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # instantiate the model
        self.model = TRANSFORMERCell(cfg)

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.scheduler.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.scheduler.lr * cfg.scheduler.lr_decay,
        )
        self.scaler = GradScaler()

        # load checkpoint
        self.epoch_init = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
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
            self.optimizer.step()

    def train(self, graph):
        """
        Perform one training iteration over one graph. The training is performed
        over multiple timesteps, where the number of timesteps is specified in
        the 'stride' parameter.

        Arguments:
            graph: the desired graph.

        Returns:
            loss: loss value.

        """
        graph = graph.to(self.device)
        self.optimizer.zero_grad()

        graph.edata["efeatures"] = graph.edata["efeatures"].squeeze()
        for istride in range(self.stride - 1):
            ns = graph.ndata["nfeatures"][:, :, istride]
            graph.ndata["current_state"] = ns
            pred = self.model(graph)
            loss = mse(pred, ns[:,0:2], mask = 1)
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

    # training loop
    start = time.time()
    logger.info("Training started...")
    loss_vector = []  # Initialize an empty list to store loss values
    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        for graph in trainer.dataloader:
            loss = trainer.train(graph)
        loss_vector.append(
            loss.cpu().detach().numpy()
        )  # Append the loss value to the vector

        if torch.cuda.is_available():
            max_memory_allocated = torch.cuda.max_memory_allocated()
        else:
            max_memory_allocated = psutil.virtual_memory()[3]

        logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}, memory allocated: {max_memory_allocated/1024**3:.2f} GB"
        )

        if cfg.training.output_interval != -1:
            if (epoch % cfg.training.output_interval) == 0 or epoch == 0 or epoch == (cfg.training.epochs-1): 
                # save checkpoint
                save_checkpoint(
                    os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
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
            with open(cfg.checkpoints.ckpt_path + "/parameters.json", "w") as outf:
                json.dump(trainer.params, outf, default=default, indent=4)
    logger.info("Training completed!")

    # Plot loss_vector
    plt.figure()
    ax = plt.axes()
    ax.semilogy(loss_vector, label="loss")
    ax.legend()
    plt.savefig("loss.png", bbox_inches="tight")

    ep, eq = evaluate_model(cfg, logger, trainer.model, trainer.params, trainer.graphs)
    return (ep + eq) / 2




@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    do_training(cfg, dist)

if __name__ == "__main__":
    main()