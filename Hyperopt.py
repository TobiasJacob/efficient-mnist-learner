import os

import numpy as np
import pandas as pd
from ax import optimize
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import init_notebook_plotting, render
from omegaconf import DictConfig

from eml.Config import Config
from eml.Datasets import load_data
from eml.Train import train

os.makedirs("hyperopt", exist_ok=True)
os.chdir("hyperopt")


cfg = DictConfig(Config(1, 1))
cfg.num_train_labels = 500
data_loaders = load_data(cfg)


def evaluation_function(params) -> None:
    cfg.unsupervised_epochs = params["unsupervised_epochs"]
    cfg.classifier_epochs = params["classifier_epochs"]
    cfg.autoencoder_features = params["autoencoder_features"]
    # cfg.autoencoder_lr = params["autoencoder_lr"]
    # cfg.classifier_lr = params["classifier_lr"]
    # cfg.classifier_lr_autoenc = params["classifier_lr_autoenc"]
    cfg.classifier_lr_autoenc = params["lr_ratio"] * cfg.classifier_lr
    cfg.auto_encoder_channels = [[16, 32, 64], [32, 64], [16, 32]][
        params["auto_encoder_channels"]
    ]
    cfg.auto_encoder_depth = params["auto_encoder_depth"]
    cfg.auto_encoder_fc_layers = params["auto_encoder_fc_layers"]
    cfg.classifier_size = params["classifier_size"]
    cfg.variational_sigma = params["variational_sigma"]
    cfg.dropout_p = params["dropout_p"]
    cfg.num_train_labels = 5000
    # cfg.classifier_neurons = [[128, 64, 32], [128, 64], [256, 128, 64], [256, 128, 64, 32]][params["classifier_neurons"]]
    result = train(cfg, *data_loaders)
    return {"accuracy": result["classifier/val_acc"]}


best_parameters, best_values, experiment, model = optimize(
    parameters=[
        {
            "name": "unsupervised_epochs",
            "type": "range",
            "bounds": [7, 20],
            "value_type": "int",
        },
        {
            "name": "classifier_epochs",
            "type": "range",
            "bounds": [10, 40],
            "value_type": "int",
        },
        {
            "name": "autoencoder_features",
            "type": "range",
            "bounds": [32, 1024],
            "value_type": "int",
            "log_scale": True,
        },
        {
            "name": "lr_ratio",
            "type": "range",
            "bounds": [0.01, 1.0],
            "value_type": "float",
            "log_scale": True,
        },
        {
            "name": "variational_sigma",
            "type": "range",
            "bounds": [0.01, 0.1],
            "value_type": "float",
            "log_scale": True,
        },
        {"name": "auto_encoder_channels", "type": "choice", "values": [0, 1, 2]},
        {
            "name": "auto_encoder_depth",
            "type": "range",
            "bounds": [0, 4],
            "value_type": "int",
        },
        {
            "name": "auto_encoder_fc_layers",
            "type": "range",
            "bounds": [0, 4],
            "value_type": "int",
        },
        {
            "name": "classifier_size",
            "type": "range",
            "bounds": [0, 6],
            "value_type": "int",
        },
        {
            "name": "dropout_p",
            "type": "range",
            "bounds": [0.001, 0.5],
            "value_type": "float",
            "log_scale": True,
        },
    ],
    objective_name="accuracy",
    evaluation_function=evaluation_function,
    minimize=False,
    total_trials=5000,
)

with open("resultsLog.txt", "w") as f:
    f.write(str(list(experiment.trials.values())) + "\n")
    f.write(str(experiment.fetch_data().df) + "\n")
    f.write(str(best_parameters) + "\n")
