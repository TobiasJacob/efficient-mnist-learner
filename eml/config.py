from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import DictConfig


@dataclass
class Config:
    unsupervised_epochs: int = 10
    classifier_epochs: int = 5

    device: str = "cuda"
    workers: int = 4
    batch_size: int = 64

    num_train_labels: int = 60000

    autoencoder_lr: float = 1e-3
    classifier_lr: float = 1e-3
    auto_encoder_channels: List[int] = field(default_factory=lambda: [16, 32])
    auto_encoder_fc_layers: int = 3
    classifier_neurons: List[int] = field(default_factory=lambda: [256, 128, 64])
    variational_sigma: Optional[float] = 0.1


def config_description(
    current_config: DictConfig, default_config: Optional[Config] = None
) -> str:
    if default_config is None:
        default_config = Config(None, None)
    desc = ""

    for fld, val in current_config.items():
        default_value = getattr(default_config, fld)
        if val != default_value:
            desc += f" {fld}={val}"
        if type(val) == DictConfig:
            desc += f" {config_description(val, default_value)}"

    return desc.strip()
