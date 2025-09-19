import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass(kw_only=True)
class VAELoss:
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    loss: torch.Tensor


def mae_vae_loss(y_hat: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_loss_w: float = 1.) -> VAELoss:
    recon_loss = F.l1_loss(y_hat, y)
    kl_loss = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - (logvar.exp() + 1e-8)))
    return VAELoss(
        recon_loss=recon_loss,
        kl_loss=kl_loss,
        loss=recon_loss + kl_loss*kl_loss_w
    )
    # return VAELoss(
    #     recon_loss=recon_loss,
    #     kl_loss=recon_loss,
    #     loss=recon_loss
    # )
