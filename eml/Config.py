from dataclasses import dataclass, field
from typing import List, Optional, Type

import torch.nn as nn
from omegaconf import DictConfig


@dataclass
class Config:
    """Global configuration object."""

    # The number of epochs for training the autoencoder.
    unsupervised_epochs: int = 0
    # The number of epochs for training the classifier together.
    classifier_epochs: int = 300

    # The compute device
    device: str = "cuda"
    # Number of workers for the data loader. Should be CPU core count.
    workers: int = 4
    # Batch size
    batch_size: int = 128

    # How many supervised training labels the NN has access to.
    num_train_labels: int = 10000

    # Learning rate for the autoencoder
    autoencoder_lr: float = 1e-3
    # Learning rate for the classifier
    classifier_lr: float = 1e-3
    # Channel size for the images in the encoder part.
    auto_encoder_channels: List[int] = field(default_factory=lambda: [16, 64])
    # Autoencoder encoded feature size
    autoencoder_features: int = 256
    # Autoencoder depth
    auto_encoder_depth: int = 3
    # Number of fully connected layers in the encoder.
    auto_encoder_fc_layers: int = 0
    # Number of neurons per layer in the classification head.
    classifier_size: int = 3
    # Sigma used for normal distribution in the variational autoencoder.
    # None does not add noise in the autoencoder.
    variational_sigma: Optional[float] = 0.01
    # Probability for dropout layer
    dropout_p: float = 0.1
    weight_decay: float = 5e-4
    autoencoder_stride: int = 2

    advanced_initialization: bool = False
    non_linearity: str = "relu"

    # Use SAM optimizer
    use_sam: bool = False
    sam_rho: float = 2.0
    sam_adaptive: bool = True
    sam_momentum: float = 0.9

    sam_autoencoder_lr: float = 0.1
    sam_classifier_lr: float = 0.1
    sam_classifier_lr_autoenc: float = 0.1


def get_non_linearity(cfg: Config) -> Type:
    if cfg.non_linearity == "relu":
        return nn.ReLU
    if cfg.non_linearity == "leaky_relu":
        return nn.LeakyReLU
    if cfg.non_linearity == "elu":
        return nn.ELU


def config_description(
    current_config: DictConfig,
    default_config: Optional[Config] = None,
    truncate: bool = True,
) -> str:
    """Helper method that generates a decription from a config object.
    The description shows all the values that differ from the default config.

    Args:
        current_config (DictConfig): The current config object
        default_config (Optional[Config], optional): The reference config, or None to
            use the global default config object. Defaults to None.

    Returns:
        str: A description summarizing the difference between the current and the
            default config.
    """
    if default_config is None:
        default_config = Config()
    desc = ""

    for fld, val in current_config.items():
        default_value = getattr(default_config, fld)
        if type(val) == DictConfig:
            desc += f" {config_description(val, default_value)}"
        elif type(val) == float and val != default_value:
            desc += f" {fld}={val:1.2}"
        elif val != default_value:
            desc += f" {fld}={val}"

    if truncate:
        return desc.strip()[0:50]
    return desc.strip()
