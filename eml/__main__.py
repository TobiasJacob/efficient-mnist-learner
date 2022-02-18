import os

import hydra
import pytorch_lightning as pl
import torch
import torchvision
from torchvision import transforms

from eml.config import Config
from eml.networks.AutoEncoder import AutoEncoder


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
    trainer = pl.Trainer(gpus=1 if cfg.device == "cuda" else 0, max_epochs=3)
    trainer.fit(auto_encoder, train_loader, eval_loader)

    # Visualize embeddings
    embeddings = []
    all_imgs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in eval_loader:
            embeddings.append(auto_encoder(imgs))
            all_imgs.append(imgs)
            all_labels.append(labels)
    embeddings = torch.cat(embeddings)
    all_imgs = torch.cat(all_imgs)
    all_labels = torch.cat(all_labels)

    trainer.logger.experiment.add_embedding(
        embeddings,
        metadata=all_labels,
        label_img=all_imgs,
    )


if __name__ == "__main__":
    main()
