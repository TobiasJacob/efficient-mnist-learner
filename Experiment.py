import os

import pandas as pd
from omegaconf import DictConfig

from eml.Config import Config
from eml.Datasets import load_data
from eml.Train import train


def main() -> None:
    os.makedirs("eval", exist_ok=True)
    os.chdir("eval")
    results = []
    cfg = DictConfig(Config())
    data_loaders = load_data(cfg)
    for lr in [1e-3, 1e-4]:
        for variational_sigma in [None, 0.1, 1.0]:
            for classifier_epochs in [1, 2, 3]:
                for num_train_labels in [1000, 5000, 10000, 60000]:
                    for unsupervised_epochs in [1, 2, 3, 4, 5]:
                        cfg.num_train_labels = num_train_labels
                        cfg.unsupervised_epochs = unsupervised_epochs
                        cfg.classifier_epochs = classifier_epochs
                        cfg.variational_sigma = variational_sigma
                        cfg.lr = lr
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
