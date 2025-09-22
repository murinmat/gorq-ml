import torch
from torch import nn
from typing import Tuple

from gorq_ml.models.vae._base import VAEModel


class SimpleVAEModel(VAEModel):
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
        self.latent_dim = latent_dim
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

    @property
    def latent_shape(self) -> list[int]:
        return [self.latent_dim]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.fc_mu(encoded), self.fc_logvar(encoded)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out_fc: torch.Tensor = self.fc_decoder(z)
        nb = z.shape[0]
        out_fc = out_fc.view(nb, -1, self.out_fc_decoder_shape, self.out_fc_decoder_shape)
        return self.decoder(out_fc)
