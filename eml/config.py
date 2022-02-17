from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig


@dataclass
class Config:
    device: str = "cuda"
    batch_size: int = 64
    unsupervised_epochs: int = 5
    workers: int = 16


def config_description(
    current_config: DictConfig, default_config: Optional[Config] = None
) -> str:
    if default_config is None:
        desc = current_config.prefix
        default_config = Config()
    else:
        desc = ""

    for field, val in current_config.items():
        default_value = getattr(default_config, field)
        if (
            type(val) == float
            or type(val) == str
            or type(val) == int
            or type(val) == list
        ):
            if val != default_value:
                desc += f" {field}={val}"
        if type(val) == DictConfig:
            desc += config_description(val, default_value)

    return desc
