import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch import nn
from typing import Tuple
from clearml.logger import Logger

from src.models._base import BaseModel
from src.losses import mae_vae_loss


class SimpleVAEModel(nn.Module):
    def __init__(
            self,
            *,
            img_size: int,
            nc: int,
            dims: list[int],
            latent_dim: int,
            final_sigmoid: bool,
    ) -> None:
        super().__init__()
        # Encoder
        all_dims = [nc] + dims
        self.encoder = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(all_dims[idx], all_dims[idx+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(all_dims[idx+1]),
                    nn.GELU(),
                    nn.Conv2d(all_dims[idx+1], all_dims[idx+1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(all_dims[idx+1]),
                    nn.GELU(),
                ) for idx in range(len(all_dims)-1)
            ],
            nn.Flatten(),
        )
        nc_in_linear = int((img_size / (2**len(dims))) ** 2 * dims[-1])
        self.fc_mu = nn.Linear(nc_in_linear, latent_dim)
        self.fc_logvar = nn.Linear(nc_in_linear, latent_dim)

        # Decoder
        out_fc_size = img_size // 2**(len(dims))
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, out_fc_size**2 * all_dims[-1]),
            nn.GELU(),
        )
        self.out_fc_decoder_shape = out_fc_size
        self.decoder = nn.Sequential(*[
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(nc_in, nc_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(nc_out) if nc_out != nc else nn.Identity(),
                    nn.GELU() if nc_out != nc else nn.Identity(),
                ) for nc_in, nc_out in zip(all_dims[::-1][:-1], all_dims[::-1][1:])
            ],
            nn.Sigmoid() if final_sigmoid else nn.Identity(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.fc_mu(encoded), self.fc_logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out_fc: torch.Tensor = self.fc_decoder(z)
        nb = z.shape[0]
        out_fc = out_fc.view(nb, -1, self.out_fc_decoder_shape, self.out_fc_decoder_shape)
        return self.decoder(out_fc)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class SimpleVAEFaceGenModel(BaseModel):
    def __init__(
        self,
        *args,
        img_size: int,
        nc: int,
        dims: list[int],
        latent_dim: int,
        final_sigmoid: bool,
        kl_warmup_steps: int,
        kl_loss_multiplier: float = 1.,
        log_samples_interval: int | None = None,
        **kwargs
    ):
        if 'ignore_hparams' in kwargs:
            kwargs['ignore_hparams'].append('val_tensor')
        else:
            kwargs['ignore_hparams'] = ['val_tensor']
        super().__init__(*args, **kwargs)
        self.val_tensor = torch.randn(8, latent_dim, generator=torch.Generator().manual_seed(42))
        self.kl_weight = np.linspace(0, 1, kl_warmup_steps) * kl_loss_multiplier
        self.model = SimpleVAEModel(
            img_size=img_size,
            nc=nc,
            dims=dims,
            latent_dim=latent_dim,
            final_sigmoid=final_sigmoid,
        )

    def _log_sample_images(self):
        is_training = self.training
        self.train(False)
        with torch.no_grad():
            out = self.model.decode(self.val_tensor.to(self.device))
            for idx, o in enumerate(out):
                img_data: torch.Tensor = o[0]
                img_data[img_data < 0.01] = torch.nan
                to_plot = img_data.cpu().detach().numpy()
                to_plot = plt.get_cmap('nipy_spectral')(to_plot)[..., :3] # type: ignore
                Logger.current_logger().report_image(
                    title=f'generated_output',
                    series=f'{idx}',
                    iteration=self.global_step,
                    image=to_plot,
                    max_image_history=-1,
                )

        self.train(is_training)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0
        log_interval = self.hparams.get('log_samples_interval', None)
        if log_interval is not None and self.global_step % log_interval == 0:
            self._log_sample_images()

    def on_train_epoch_end(self) -> None:
        self._log_sample_images()

    def compute_losses(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        kl_weight = 1 if self.global_step >= len(self.kl_weight) else self.kl_weight[self.global_step]
        recon, mu, logvar = self.model(batch)
        loss = mae_vae_loss(batch, recon, mu, logvar, kl_loss_w=kl_weight)
        return recon, {
            'loss': loss.loss,
            'recon_loss': loss.recon_loss,
            'kl_koss': loss.kl_loss,
            'kl_loss_weight': kl_weight,
        }
