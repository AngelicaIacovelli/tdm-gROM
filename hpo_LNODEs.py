import torch
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch
import hydra
from omegaconf import DictConfig
from train_LNODEs import do_training
import os
from modulus.distributed.manager import DistributedManager
import random
import numpy as np

# initialize distributed manager
DistributedManager.initialize()
dist = DistributedManager()

def objective(config, cfg):  
    cfg.checkpoints.ckpt_path = os.getcwd() + "/" + cfg.checkpoints.ckpt_path 
    
    cfg.LNODEs_architecture.lr = config["lr"] 
    cfg.LNODEs_architecture.lr_decay = config["lr_decay"]
    cfg.LNODEs_architecture.N_states = config["N_states"]
    cfg.LNODEs_architecture.N_hid_MLP = config["N_hid_MLP"]
    cfg.LNODEs_architecture.N_neu_MLP = config["N_neu_MLP"]

    metric = do_training(cfg, dist).cpu().detach().numpy()
    
    train.report({"Average_error": float(metric)})  # Report to Tune

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg.work_directory = os.getcwd()
    cfg.training.output_interval = cfg.training.epochs - 1 

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "lr_decay": tune.loguniform(1e-3, 1e-1), 
        "N_states": tune.randint(5472, 10000),
        "N_hid_MLP": tune.randint(1, 50),
        "N_neu_MLP": tune.randint(1, 50),
    }
    algo = OptunaSearch()  

    def objective_cfg(config):

        return objective(config, cfg)

    objective_with_gpu = tune.with_resources(objective_cfg, {"gpu": 1})

    storage_path = os.path.expanduser("/home/aiacovelli/ray_results")
    exp_name = "hpo_LNODEs_experiment"
    path = os.path.join(storage_path, exp_name)

    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(
            path, 
            trainable = objective_with_gpu, 
            param_space=search_space,
            resume_errored=True
        )
    else:
        tuner = tune.Tuner(  
            trainable = objective_with_gpu,
            tune_config=tune.TuneConfig(
                metric="Average_error", 
                mode="min", 
                search_alg=algo,
                num_samples=cfg.hyperparameter_optimization.runs
            ),
            run_config=train.RunConfig(storage_path=storage_path, name=exp_name),
            param_space=search_space,
        )
    
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == "__main__":
    main()
