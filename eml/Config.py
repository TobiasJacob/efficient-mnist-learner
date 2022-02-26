from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import DictConfig


@dataclass
class Config:
    """Global configuration object."""

    # The number of epochs for training the autoencoder.
    unsupervised_epochs: int = 10
    # The number of epochs for training the classifier together.
    classifier_epochs: int = 10

    # The compute device
    device: str = "cuda"
    # Number of workers for the data loader. Should be CPU core count.
    workers: int = 4
    # Batch size
    batch_size: int = 64

    # How many supervised training labels the NN has access to.
    num_train_labels: int = 60000

    # Learning rate for the autoencoder
    autoencoder_lr: float = 1e-3
    # Learning rate for the classifier
    classifier_lr: float = 1e-3
    # Channel size for the images in the encoder part.
    auto_encoder_channels: List[int] = field(default_factory=lambda: [16, 32])
    # Number of fully connected layers in the encoder.
    auto_encoder_fc_layers: int = 1
    # Number of neurons per layer in the classification head.
    classifier_neurons: List[int] = field(default_factory=lambda: [256, 128, 64])
    # Sigma used for normal distribution in the variational autoencoder.
    # None does not add noise in the autoencoder.
    variational_sigma: Optional[float] = 0.01


def config_description(
    current_config: DictConfig, default_config: Optional[Config] = None
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
        if val != default_value:
            desc += f" {fld}={val}"
        if type(val) == DictConfig:
            desc += f" {config_description(val, default_value)}"

    return desc.strip()
