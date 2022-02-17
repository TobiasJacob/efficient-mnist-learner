import os

import hydra
import torch
import torchvision
from omegaconf import OmegaConf
from torchvision import transforms

from eml.config import Config
from eml.networks.AutoEncoderClassifier import AutoEncoderClassifier


@hydra.main(config_path=None)
def my_app(cfg: Config) -> None:
    torch.zeros((10, 10), device="cuda")
    dataset = torchvision.datasets.FashionMNIST(
        os.path.expanduser("~/dataset"),
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    print(len(dataset))
    orig_train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True
    )

    nn = AutoEncoderClassifier(image_size=(28, 28))
    imgs, labels = next(iter(orig_train_loader))
    print(nn(imgs).shape)
    print(dataset[0][0].shape)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
