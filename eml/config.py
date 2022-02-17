from dataclasses import dataclass


@dataclass
class Config:
    device: str = "cuda"
