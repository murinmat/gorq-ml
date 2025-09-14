import numpy as np
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer


class LinearWarmupLR(LRScheduler):

    def __init__(self, optimizer: Optimizer, warmup_steps: int, **kwargs):
        self._lr_arr = np.linspace(0, 1, warmup_steps + 1)
        self._warmup_steps = warmup_steps
        self._final_lrs = [g['lr'] for g in optimizer.param_groups]
        super().__init__(optimizer, **kwargs)

    def get_lr(self) -> list[float]:
        fraction = self._lr_arr[self._step_count] if self._step_count <= self._warmup_steps else 1.
        return [fraction * final_lr for final_lr in self._final_lrs]
