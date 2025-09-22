import torch
import abc
from torch import nn
from typing import Tuple


class VAEModel(nn.Module, abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @abc.abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def latent_shape(self) -> list[int]:
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
