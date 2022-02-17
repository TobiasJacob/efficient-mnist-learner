import os
import hydra
from omegaconf import OmegaConf
import torch

from eml.config import Config
import torchvision


@hydra.main(config_path=None)
def my_app(cfg: Config) -> None:
    torch.zeros((10, 10), device="cuda")
    dataset = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"), train=False, download=True
    )
    print(dataset[0])
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
