import os

import hydra
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

from eml.callbacks.VisualizeEmbeddings import VisualizeEmbeddings
from eml.config import Config
from eml.networks.AutoEncoder import AutoEncoder
from eml.networks.Classifier import Classifier


@hydra.main(config_path=None)
def main(cfg: Config) -> None:
    cfg = Config()
    print(cfg)
    # writer = SummaryWriter(config_description(cfg, Config()))
    train_dataset = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.workers
    )
    eval_dataset = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"),
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.workers
    )

    auto_encoder = AutoEncoder((28, 28))
    trainer_autoencoder = pl.Trainer(
        gpus=1 if cfg.device == "cuda" else 0,
        max_epochs=3,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            VisualizeEmbeddings(num_batches=15),
            LearningRateMonitor("epoch"),
        ],
    )
    # trainer_autoencoder.fit(auto_encoder, train_loader, eval_loader)
    classifier = Classifier(auto_encoder)
    trainer_classifier = pl.Trainer(
        gpus=1 if cfg.device == "cuda" else 0,
        max_epochs=3,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
    )
    classifier.auto_encoder.freeze()
    trainer_classifier.fit(classifier, train_loader, eval_loader)


if __name__ == "__main__":
    main()
