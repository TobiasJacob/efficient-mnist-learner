import os

import pandas as pd
from omegaconf import DictConfig

from eml.Config import Config
from eml.Datasets import load_data
from eml.Train import train


def main() -> None:
    """Runs a series of different configurations to figure out the ideal parameters"""
    os.makedirs("eval", exist_ok=True)
    os.chdir("eval")
    results = []
    cfg = DictConfig(Config())
    for num_train_labels in [500, 1000, 2000, 5000, 10000, 60000]:
        cfg.num_train_labels = num_train_labels
        data_loaders = load_data(cfg)
        for variational_sigma in [0.01]:
            for unsupervised_epochs in [7]:
                classifier_epochs = 25000 // num_train_labels + 7
                cfg.unsupervised_epochs = unsupervised_epochs
                cfg.classifier_epochs = classifier_epochs
                cfg.variational_sigma = variational_sigma
                result = train(cfg, *data_loaders)
                results.append(
                    (
                        num_train_labels,
                        unsupervised_epochs,
                        classifier_epochs,
                        result["classifier/val_acc"],
                        result["classifier/val_f1"],
                    )
                )
                df = pd.DataFrame(
                    results,
                    columns=[
                        "num_train_labels",
                        "unsupervised_epochs",
                        "classifier_epochs",
                        "val_acc",
                        "val_f1",
                    ],
                )
                print(df)
                df.to_csv("eval.csv")


if __name__ == "__main__":
    main()
