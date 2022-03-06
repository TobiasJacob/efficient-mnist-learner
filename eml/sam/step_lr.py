import torch
from torch.optim.lr_scheduler import _LRScheduler


class StepLR(_LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, learning_rate: float, total_epochs: int
    ) -> None:
        self.last_epoch = 0
        self.total_epochs = total_epochs
        self.base = learning_rate
        self.optimizer = optimizer
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return [self.base] * len(self.optimizer.param_groups)
        if self.last_epoch < self.total_epochs * 5 / 10:
            lr = self.base
        elif self.last_epoch < self.total_epochs * 8 / 10:
            lr = self.base * 0.2
        elif self.last_epoch < self.total_epochs * 9 / 10:
            lr = self.base * 0.2**2
        else:
            lr = self.base * 0.2**3
        return [lr] * len(self.optimizer.param_groups)
