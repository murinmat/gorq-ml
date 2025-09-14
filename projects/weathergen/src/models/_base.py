import torch
import abc
from torch import nn
from typing import Tuple, Type
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from lightning import LightningModule


class WeatherGenModel(abc.ABC, LightningModule):
    def __init__(
        self,
        *,
        lr: float,
        betas: tuple[float, float],
        lr_warmup_steps: int,
        total_training_steps: int,
        ignore_hparams: list[str],
        val_metrics: dict[str, nn.Module] = dict(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=ignore_hparams)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                self.parameters(),
                self.hparams['lr'],
                betas=self.hparams['betas'],
            ),
        ]
        lr_schedulers = [
            {
                'scheduler': SequentialLR(
                        optimizer=optimizers[0],
                        schedulers=[
                            LinearLR(
                                optimizer=optimizers[0],
                                start_factor=1e-5,
                                end_factor=1.,
                                total_iters=self.hparams['lr_warmup_steps'],
                            ),
                            CosineAnnealingLR(
                                optimizer=optimizers[0],
                                T_max=self.hparams['total_training_steps'] - self.hparams['lr_warmup_steps'],
                            )
                        ],
                        milestones=[self.hparams['lr_warmup_steps']]
                    ),
                'interval': 'step',
        }]
        return optimizers, lr_schedulers

    @abc.abstractmethod
    def compute_losses(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        ...

    def training_step(self, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        loss_dict = self.compute_losses(batch)[1]
        for k, v in loss_dict.items():
            self.log(
                name=f'{k}/train',
                value=v,
                prog_bar=k=='loss',
                on_step=True,
                on_epoch=True,
                enable_graph=k=='loss',
                sync_dist=True,
            )
        return loss_dict['loss'] # type: ignore

    def validation_step(self, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output, loss_dict = self.compute_losses(batch)
        for k, v in loss_dict.items():
            self.log(
                name=f'{k}/val',
                value=v,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        for k, v in self.hparams['val_metrics'].items():
            value = v(output, batch)
            self.log(
                name=f'{k}/val',
                value=value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return output
