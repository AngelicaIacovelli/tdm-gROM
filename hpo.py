import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import hydra
from omegaconf import DictConfig
from train import do_training
import os

def objective(config, cfg):  
    cfg.scheduler.lr = config["lr"] 
    cfg.scheduler.lr_decay = config["lr_decay"]
    cfg.training.batch_size = config["batch_size"]
    cfg.training.loss_weight_boundary_nodes = config["loss_weight_boundary_nodes"]
    cfg.architecture.hidden_dim = config["hidden_dim"]
    cfg.architecture.latent_size_gnn = config["latent_size_gnn"]
    cfg.architecture.latent_size_mlp = config["latent_size_mlp"]
    cfg.architecture.number_hidden_layers_mlp = config["number_hidden_layers_mlp"]
    cfg.architecture.autoloop_iterations = config["autoloop_iterations"]

    metric = do_training(cfg)
    train.report({"inference_performance": metric})  # Report to Tune

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    cfg.work_directory = os.getcwd()
    cfg.checkpoints.ckpt_path = os.getcwd() + "/" + cfg.checkpoints.ckpt_path

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "lr_decay": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.randint(10, 200), 
        "loss_weight_boundary_nodes": tune.randint(1, 200), 
        "hidden_dim": tune.randint(1, 64), 
        "latent_size_gnn": tune.randint(1, 64),
        "latent_size_mlp": tune.randint(1, 200),
        "number_hidden_layers_mlp": tune.randint(1, 50),
        "autoloop_iterations": tune.randint(1, 3),

    }
    algo = OptunaSearch()  

    def objective_cfg(config):
        return objective(config, cfg)

    tuner = tune.Tuner(  
        objective_cfg,
        tune_config=tune.TuneConfig(
            metric="inference_performance", mode="min", search_alg=algo
        ),
        run_config=train.RunConfig(stop={"training_iteration": 1}),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

if __name__ == "__main__":
    main()
