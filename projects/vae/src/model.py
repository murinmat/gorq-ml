import torch
import numpy as np
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset
from typing import Tuple

from gorq_ml.models.lightning._base import BaseModel
from gorq_ml.models import vae
from gorq_ml.training.utils import log_image


class VAELightningModel(BaseModel):
    def __init__(
        self,
        *args,
        model_name: str,
        model_kwargs: dict,
        kl_warmup_steps: int,
        kl_loss_multiplier: float = 1.,
        log_samples_interval: int | None = None,
        plot_colormap: str | None = None,
        plot_per_channel: bool = False,
        val_dataset: Dataset | None,
        log_val_indices: list[int] = [],
        **kwargs
    ):
        if 'ignore_hparams' in kwargs:
            kwargs['ignore_hparams'].append('val_tensor')
        else:
            kwargs['ignore_hparams'] = ['val_tensor']
        kwargs['ignore_hparams'].append('val_dataset')
        if len(log_val_indices) == 0 and val_dataset is None:
            raise ValueError(f'Log val indices is not empty but no dataset was provided.')
        super().__init__(*args, **kwargs)
        self.model: vae.VAEModel = getattr(vae, model_name)(
            **model_kwargs
        )
        self.val_tensor = torch.randn(8, *self.model.latent_shape, generator=torch.Generator().manual_seed(42))
        self.kl_weight = np.linspace(0, 1, kl_warmup_steps) * kl_loss_multiplier
        self.val_dataset = val_dataset

    @torch.no_grad()
    def _log_sample_images(self):
        is_training = self.training
        self.train(False)
        colormap = self.hparams.get('plot_colormap', None)

        to_plot = {
            'generated': self.model.decode(self.val_tensor.to(self.device)),
        }
        if self.val_dataset is not None:
            to_plot['reconstructed'] = torch.cat([
                self.model(self.val_dataset[x].to(self.device)[None])[0]
                for x in self.hparams['log_val_indices']
            ], dim=0)
            
        
        for subset, outputs in to_plot.items():
            logger.critical(f'Logging {subset} of shape {outputs.shape}')
            for idx, output in enumerate(outputs):
                if self.hparams['plot_per_channel']:
                    for c_out in output:
                        log_image(
                            data=c_out,
                            title=subset,
                            series=f'Index {idx}',
                            iteration=self.global_step,
                            max_history=-1,
                            colormap=colormap,
                        )
                else:
                    log_image(
                        data=output[0] if colormap is not None else output.permute(1, 2, 0),
                        title=subset,
                        series=f'Index {idx}',
                        iteration=self.global_step,
                        max_history=-1,
                        colormap=colormap,
                    )
            self.train(is_training)

    def loss_fn(
            self,
            input: torch.Tensor,
            reconstruction: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            kl_loss_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(reconstruction, input)
        kldiv_loss = - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss_weight * kldiv_loss, recon_loss, kldiv_loss

    def on_train_batch_end(self, *args, **kwargs) -> None:
        log_interval = self.hparams.get('log_samples_interval', None)
        if log_interval is not None and self.global_step % log_interval == 0:
            self._log_sample_images()

    def on_train_epoch_end(self) -> None:
        self._log_sample_images()

    def compute_losses(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        kl_weight = 1 if self.global_step >= len(self.kl_weight) else self.kl_weight[self.global_step]
        recon, mu, logvar = self.model(batch)
        loss, recon_loss, kldiv_loss = self.loss_fn(batch, recon, mu, logvar, kl_loss_weight=kl_weight)
        return recon, {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_koss': kldiv_loss,
            'kl_loss_weight': kl_weight,
        }
