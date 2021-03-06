from typing import List, Type

import torch
import torch.nn as nn

from eml.networks.BasicUnit import BasicUnit
from eml.networks.FCUnit import FCUnit
from eml.networks.UpsampleUnit import UpsampleUnit


class Decoder(nn.Module):
    """Generates images from a feature vector using unflattening and transpoed
    convolutions.
    """

    def __init__(
        self,
        encoded_feature_size: int,
        fc_size: int,
        num_fc_layers: int,
        channels: List[int],
        dropout_p: float,
        depth: int,
        non_linearity: Type,
        stride: int,
    ) -> None:
        """Creates a new Decoder Module.

        Args:
            channels (List[int], optional): The channel sizes for the convolutions.
            dropout_p (float): Probability for dropout layer.
        """
        super().__init__()

        # Fully connected part
        fc_layers = []
        fc_layers.append(nn.Linear(encoded_feature_size, fc_size))
        for _ in range(num_fc_layers):
            fc_layers.append(FCUnit(fc_size, dropout_p, non_linearity))

        self.fc_layers = nn.Sequential(*fc_layers)

        decoder = []
        for i in reversed(range(len(channels))):
            out_features = 1 if i == 0 else channels[i - 1]
            for _ in range(depth):
                decoder.append(BasicUnit(channels[i], dropout_p, non_linearity))
            decoder.append(
                UpsampleUnit(
                    channels[i], out_features, stride, dropout_p, non_linearity
                )
            )
        self.decoder = nn.Sequential(*decoder)

    def forward(
        self,
        x: torch.Tensor,
        orig_shape_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the foward pass.

        Args:
            x (torch.Tensor): The embedded features. Shape: (batch_size, fc_size)
            pool_indices (List[torch.Tensor]): The pooling indices from the Encoder.
            layer_sizes (List[torch.Tensor]): The layer sizes from the Encoder.
            orig_shape_2d (torch.Tensor): The unflattened feature shape from the
                Encoder.

        Returns:
            torch.Tensor: The reconstructed images
        """
        x = self.fc_layers(x)
        x = x.reshape(orig_shape_2d)
        x = self.decoder(x)
        return x
