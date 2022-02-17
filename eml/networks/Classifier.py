import torch
import torch.nn as nn

from eml.networks.AutoEncoder import AutoEncoder


class Classifier(nn.Module):
    def __init__(
        self,
        auto_encoder: AutoEncoder,
        fc_size: int,
        output_classes: int = 10,
    ) -> None:
        super().__init__()
        self.auto_encoder = auto_encoder

        # Classifier
        self.classifier = [
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Linear(fc_size, output_classes),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.auto_encoder(x)
        for layer in self.classifier:
            x = layer(x)
        return x
