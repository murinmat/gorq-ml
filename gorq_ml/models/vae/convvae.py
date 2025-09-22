import torch
import numpy as np
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch import nn
from typing import Tuple
from clearml.logger import Logger

from gorq_ml.models.vae._base import VAEModel


class ConvVAEModel(VAEModel):
    def __init__(
            self,
            *,
            img_size: int,
            nc: int,
            dims: list[int],
            latent_nc: int,
            final_sigmoid: bool,
    ) -> None:
        super().__init__()
        # Encoder
        self.latent_hw = img_size // (2**(len(dims)))
        self.latent_nc = latent_nc
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
        )
        self.conv_mu = nn.Conv2d(all_dims[-1], latent_nc, 1)
        self.conv_logvar = nn.Conv2d(all_dims[-1], latent_nc, 1)
        self.conv_to_latent = nn.Conv2d(latent_nc, all_dims[-1], 1)

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
        return self.conv_mu(encoded), self.conv_logvar(encoded)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_as_latent = self.conv_to_latent(z)
        return self.decoder(z_as_latent)

    @property
    def latent_shape(self) -> list[int]:
        return [self.latent_nc, self.latent_hw, self.latent_hw]        
