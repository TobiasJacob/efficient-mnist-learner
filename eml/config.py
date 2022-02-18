from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import DictConfig


@dataclass
class Config:
    device: str = "cuda"
    workers: int = 16
    batch_size: int = 64

    num_train_labels: int = 60000

    unsupervised_epochs: int = 10
    classifier_epochs: int = 10

    lr: float = 1e-3
    channels: List[int] = field(default_factory=lambda: [6, 10, 20])


def config_description(
    current_config: DictConfig, default_config: Optional[Config] = None
) -> str:
    if default_config is None:
        default_config = Config()
    else:
        desc = ""

    for fld, val in current_config.items():
        default_value = getattr(default_config, fld)
        if (
            type(val) == float
            or type(val) == str
            or type(val) == int
            or type(val) == list
        ):
            if val != default_value:
                desc += f" {fld}={val}"
        if type(val) == DictConfig:
            desc += f" {config_description(val, default_value)}"

    return desc.strip()
