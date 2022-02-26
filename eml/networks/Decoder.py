from typing import List

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Generates images from a feature vector using unflattening and transpoed
    convolutions.
    """

    def __init__(
        self,
        fc_size: int,
        num_fc_layers: int,
        channels: List[int],
        dropout_p: float,
    ) -> None:
        """Creates a new Decoder Module.

        Args:
            channels (List[int], optional): The channel sizes for the convolutions.
            dropout_p (float): Probability for dropout layer.
        """
        super().__init__()

        # Fully connected part
        fc_layers = []
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(fc_size, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(fc_size))
            fc_layers.append(nn.Dropout(dropout_p))
        if num_fc_layers > 0:
            fc_layers.append(nn.Linear(fc_size, fc_size))

        self.fc_layers = nn.ModuleList(fc_layers)

        decoder = []
        for i in reversed(range(len(channels))):
            out_features = 1 if i == 0 else channels[i - 1]
            decoder.append(nn.MaxUnpool2d((2, 2)))
            decoder.append(nn.ConvTranspose2d(channels[i], out_features, 3))
            if i != 0:
                decoder.append(nn.ReLU())
                decoder.append(nn.BatchNorm2d(out_features))
                decoder.append(nn.Dropout2d(dropout_p))
        self.decoder = nn.ModuleList(decoder)

    def forward(
        self,
        x: torch.Tensor,
        pool_indices: List[torch.Tensor],
        layer_sizes: List[torch.Tensor],
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
        for layer in self.fc_layers:
            x = layer(x)
        x = x.reshape(orig_shape_2d)
        for layer in self.decoder:
            if type(layer) is nn.MaxUnpool2d:
                x = layer(x, pool_indices.pop(), layer_sizes.pop())
            else:
                x = layer(x)
        return x
