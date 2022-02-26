from typing import List, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes an image into a feature vector."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_fc_layers: int,
        channels: List[int],
    ) -> None:
        """Creates a new encoder module. The encoder applies convolutions and
        max-pooling first. Then, the features are flattend and processed with
        fully connected layers.

        Args:
            image_size (Tuple[int, int]): The size of the input image.
            num_fc_layers (int): Number of fully connected layers after flattening.
            channels (List[int]): Channel size of the convolutional layers.
        """
        super().__init__()
        # Down convolutions
        encoder = []
        for i in range(len(channels)):
            encoder.append(nn.Conv2d(1 if i == 0 else channels[i - 1], channels[i], 3))
            encoder.append(nn.ReLU())
            encoder.append(nn.BatchNorm2d(channels[i]))
            encoder.append(nn.MaxPool2d((2, 2), return_indices=True))
        self.encoder = nn.ModuleList(encoder)

        # Fully connected part
        x = torch.zeros((1, 1, *image_size))
        x = self(x, simulate=True)
        self.fc_size = x.flatten().shape[0]
        fc_layers = []
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.fc_size, self.fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(self.fc_size))
        if num_fc_layers > 0:
            fc_layers.append(nn.Linear(self.fc_size, self.fc_size))

        self.fc_layers = nn.ModuleList(fc_layers)
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
        pool_indices = []
        layer_sizes = []
        for layer in self.encoder:
            if type(layer) is nn.MaxPool2d:
                layer_sizes.append(x.shape)
                x, ind = layer(x)
                pool_indices.append(ind)
            else:
                x = layer(x)
        orig_shape_2d = x.shape
        # When simulating, return the x for calculating self.fc_size in the constructor
        if simulate:
            return x
        # Flatten and process with fully connected layers
        x = x.reshape(x.shape[0], self.fc_size)
        for layer in self.fc_layers:
            x = layer(x)
        return x, pool_indices, layer_sizes, orig_shape_2d
