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
import time, os, random
import numpy as np
import hydra

from transformer import TransformerCell

import generate_dataset as gd
from generate_dataset import generate_normalized_graphs
from generate_dataset import train_test_split
from generate_dataset import Bloodflow1DDataset

import json
from omegaconf import DictConfig
import uuid
import os
from torch.autograd import Function

from torch.utils.data import DataLoader

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

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Set the random seed
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Parameters.
    N_t = 41
    N_lat = 60
    N_inn = 10 #hpo
    N_g = 4 # hpo
    N_mu = 30
    N_neu_MLP_p = 30 # hpo
    N_hid_MLP_p = 3  # hpo
    N_neu_MLP_m = 30 # hpo
    N_hid_MLP_m = 3  # hpo
    
    # instantiate the model
    model = TransformerCell(cfg)

    # instantiate loss, optimizer, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.scheduler.lr * cfg.scheduler.lr_decay,
    )

    # simple dataset
    mu  = torch.randn((cfg.training.batch_size, cfg.transformer_architecture.N_timesteps))
    z_0 = torch.randn((cfg.training.batch_size, cfg.architecture.latent_size_AE))
    Z   = torch.randn((cfg.training.batch_size, cfg.transformer_architecture.N_timesteps, cfg.architecture.latent_size_AE))
    Z[:, 0, :] = z_0

    start_time = time.time()

    for epoch in range(cfg.training.epochs):
        # enable train mode
        model.train()

        # Incremental loss function
        for idx_t in range(2, cfg.transformer_architecture.N_timesteps):
            loss = cfg.transformer_architecture.threshold + 1.
            while (loss > cfg.transformer_architecture.threshold):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                Z_tilde = model(mu, z_0, idx_t)

                # Calculate the loss
                loss = mse(Z_tilde[:, 2:, :], Z[:, 1 : idx_t, :], mask = 1)

                # Backpropagation and optimization
                loss.backward()

                # Update optimizer
                optimizer.step()

                print(f'Epoch {epoch + 1}, Increment: {idx_t}, Loss: {loss.item()}')

        # Adjust learning rate using the scheduler
        scheduler.step()

    print(f"Elapsed time: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    main()