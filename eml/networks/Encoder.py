from typing import List, Tuple, Type

import torch
import torch.nn as nn

from eml.networks.BasicUnit import BasicUnit
from eml.networks.DownsampleUnit import DownsampleUnit
from eml.networks.FCUnit import FCUnit


class Encoder(nn.Module):
    """Encodes an image into a feature vector."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_fc_layers: int,
        channels: List[int],
        dropout_p: float,
        depth: int,
        encoded_feature_size: int,
        non_linearity: Type,
    ) -> None:
        """Creates a new encoder module. The encoder applies convolutions and
        max-pooling first. Then, the features are flattend and processed with
        fully connected layers.

        Args:
            image_size (Tuple[int, int]): The size of the input image.
            num_fc_layers (int): Number of fully connected layers after flattening.
            channels (List[int]): Channel size of the convolutional layers.
            dropout_p (float): Probability for dropout layer.
        """
        super().__init__()
        # Down convolutions
        encoder = []
        for i in range(len(channels)):
            in_channels = 1 if i == 0 else channels[i - 1]
            for _ in range(depth):
                encoder.append(BasicUnit(in_channels, dropout_p, non_linearity))
            encoder.append(
                DownsampleUnit(in_channels, channels[i], 3, dropout_p, non_linearity)
            )
        self.encoder = nn.Sequential(*encoder)

        # Fully connected part
        x = torch.zeros((1, 1, *image_size))
        x = self(x, simulate=True)
        self.fc_size = x.flatten().shape[0]
        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(FCUnit(self.fc_size, dropout_p, non_linearity))
        fc_layers.append(nn.Linear(self.fc_size, encoded_feature_size))

        self.fc_layers = nn.Sequential(*fc_layers)
        print(f"Encoded feature size: {self.fc_size}")

    def forward(
        self, x: torch.Tensor, simulate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies forward pass of the encoder. If simulate is True, it will only
        return the tensor after the convolutional part. This is used while building the
        network to calculate the size of the encoded features that will fed into the
        fully connected layers.

        Args:
            x (torch.Tensor): The images to encode. Shape: (batch_size, width, height).
            simulate (bool, optional): Simulation mode active. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Returns a
            tuple containing the encoded features, the pooling indices, the layer sizes
            and the unflattend 2d shape for the decoder.
                Encoded feature shape: (batch_size, self.fc_size)
        """
        # Apply down convolutions
        x = self.encoder(x)
        orig_shape_2d = x.shape
        # When simulating, return the x for calculating self.fc_size in the constructor
        if simulate:
            return x
        # Flatten and process with fully connected layers
        x = x.reshape(x.shape[0], self.fc_size)
        x = self.fc_layers(x)
        return x, orig_shape_2d
