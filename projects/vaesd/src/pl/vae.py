import torch
import torch.nn.functional as F
import lightning as L
from loguru import logger
from torch.utils.data import Dataset
from typing import Literal, Tuple
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from src.models.vae import VAE
from src.models.vqvae import VQVAE
from src.models.discriminator import Discriminator

from gorq_ml.training.utils import log_image


class VAELightning(L.LightningModule):
    def __init__(
            self,
            model_name: Literal['VAE', 'VQVAE'],
            model_kwargs: dict,
            discriminator_kwargs: dict,
            model_lr: float,
            model_betas: Tuple[float, float],
            disc_lr: float,
            disc_betas: Tuple[float, float],
            disc_step_start: int,
            codebook_weight: float,
            commitment_beta: float,
            disc_weight: float,
            perceptual_weight: float,
            viz_val_indices: list[int],
            viz_train_indices: list[int],
            viz_frequency: int,
            val_dataset: Dataset | None = None,
            train_dataset: Dataset | None = None,
            plot_colormap: str | None = None,
            plot_per_channel: bool = False,
    ) -> None:
        super().__init__()
        self.lpips = LPIPS(net_type='vgg').eval()
        if model_name not in ['VQVAE']:
            raise ValueError(f'Unknown model name: {model_name}')
        self.model: VAE | VQVAE = globals()[model_name](**model_kwargs)
        self.discriminator = Discriminator(**discriminator_kwargs)
        self.save_hyperparameters(ignore=['lpips', 'train_dataset', 'val_dataset'])
        self.automatic_optimization = False
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset
        self.viz_val_indices = viz_val_indices
        self.viz_train_indices = viz_train_indices

    def get_generator_loss(
            self,
            inp: torch.Tensor,
            out: torch.Tensor,
            quantize_losses: dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_dict = {}
        final_loss = 0

        # Pixel-wise loss
        recon_loss = F.mse_loss(out, inp)
        loss_dict['recon_loss'] = recon_loss.item()
        final_loss += recon_loss
        # VQVAE loss (codebook + commitment losses)
        codebook_loss = self.hparams['codebook_weight'] * quantize_losses['codebook_loss']
        loss_dict['codebook_loss'] = codebook_loss.item()
        final_loss += codebook_loss
        commitment_loss = self.hparams['commitment_beta'] * quantize_losses['commitment_loss']
        loss_dict['commitment_loss'] = commitment_loss.item()
        final_loss += commitment_loss
        # Perceptual loss (LPIPS)
        out_for_lpips = out.clamp(-1, 1).float()
        inp_for_lpips = inp.clamp(-1, 1).float()
        if out_for_lpips.shape[-3] == 1:
            out_for_lpips = torch.cat([out_for_lpips]*3, dim=-3)
            inp_for_lpips = torch.cat([inp_for_lpips]*3, dim=-3)
        perceptual_loss = self.hparams['perceptual_weight'] * self.lpips(out_for_lpips, inp_for_lpips).mean()
        loss_dict['perceptual_loss'] = perceptual_loss.item()
        final_loss += perceptual_loss
        # Discriminator only if the training of it should have started by now
        if self.global_step//2 > self.hparams['disc_step_start']:
            disc_fake_pred = self.discriminator(out)
            gen_disc_loss = self.hparams['disc_weight'] * F.mse_loss(disc_fake_pred, torch.ones_like(disc_fake_pred))
            loss_dict['gen_disc_loss'] = gen_disc_loss.item()
            final_loss += gen_disc_loss
        return final_loss, loss_dict

    def configure_optimizers(self) -> list[Optimizer]:
        optim_ae = Adam(
            self.model.parameters(),
            lr=self.hparams['model_lr'],
            betas=self.hparams['model_betas'],
        )
        optim_disc = Adam(
            self.discriminator.parameters(),
            lr=self.hparams['disc_lr'],
            betas=self.hparams['disc_betas']
        )
        return [optim_ae, optim_disc]
    
    def train_generator(
            self,
            inp: torch.Tensor,
            out: torch.Tensor,
            quantize_losses: dict[str, torch.Tensor]
    ) -> None:
        optim: Optimizer = self.optimizers()[0] # type: ignore

        self.toggle_optimizer(optim)
        loss, loss_dict = self.get_generator_loss(inp, out, quantize_losses)
        self.manual_backward(loss)
        optim.step()
        optim.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optim)
        self.log('gen_loss/train', loss.item(), on_step=True, on_epoch=True, logger=True)
        self.log_dict({f'{k}/train': v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)

    def get_discriminator_loss(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        disc_fake_pred = self.discriminator(fake.detach())
        disc_real_pred = self.discriminator(real)
        disc_fake_loss = self.hparams['disc_weight'] * F.mse_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = self.hparams['disc_weight'] * F.mse_loss(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        return disc_loss, {
            'disc_fake_loss': disc_fake_loss.item(),
            'disc_real_loss': disc_real_loss.item(),
        }

    def train_discriminator(
            self,
            inp: torch.Tensor,
            out: torch.Tensor,
    ) -> None:
        optim: Optimizer = self.optimizers()[1] # type: ignore

        self.toggle_optimizer(optim)
        loss, loss_dict = self.get_discriminator_loss(real=inp, fake=out)
        self.manual_backward(loss)
        optim.step()
        optim.zero_grad(set_to_none=True)
        self.untoggle_optimizer(optim)

        self.log(f'disc_loss/train', loss.item(), on_step=True, on_epoch=True)
        self.log_dict({f'{k}/train': v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)

    def training_step(self, batch: torch.Tensor) -> None:
        out, _, quantize_losses = self.model(batch)
        self.train_generator(batch, out, quantize_losses)
        if self.global_step//2 > self.hparams['disc_step_start']:
            self.train_discriminator(batch, out)
        
    def validation_step(self, batch: torch.Tensor) -> None:
        out, _, quantize_losses = self.model(batch)
        gen_loss, gen_loss_dict = self.get_generator_loss(batch, out, quantize_losses)
        disc_loss, disc_loss_dict = self.get_discriminator_loss(real=batch, fake=out)
        self.log('gen_loss/val', gen_loss.item(), on_step=False, on_epoch=True)
        self.log(f'disc_loss/val', disc_loss.item(), on_step=False, on_epoch=True)
        self.log_dict(
            {f'{k}/val': v for k, v in (gen_loss_dict | disc_loss_dict).items()},
            on_step=False,
            on_epoch=True
        )

    @torch.no_grad()
    def _log_sample_images(self, ds: Dataset | None, indices: list[int], type: str):
        is_training = self.training
        self.train(False)
        if ds is None:
            return
        colormap = self.hparams.get('plot_colormap', None)
        all_outputs = []
        for val_sample_idx in indices:
            inp = ds[val_sample_idx].to(self.device)[None]
            out = self.model(inp)[0]
            all_outputs.append(torch.cat([
                torch.clamp((inp.cpu() + 1) / 2, 0, 1),
                torch.zeros_like(inp[..., :10, :], device='cpu'),
                torch.clamp((out.cpu() + 1) / 2, 0, 1),
            ], dim=-2))
        to_plot = torch.cat(all_outputs, dim=0)

        for idx, output in enumerate(to_plot):
            if self.hparams['plot_per_channel']:
                for c_out in output:
                    log_image(
                        data=c_out,
                        title=f'Viz {type}',
                        series=f'Index {idx}',
                        iteration=self.global_step,
                        max_history=-1,
                        colormap=colormap,
                    )
            else:
                log_image(
                    data=output[0] if colormap is not None else output.permute(1, 2, 0),
                    title=f'Viz {type}',
                    series=f'Index {idx}',
                    iteration=self.global_step,
                    max_history=-1,
                    colormap=colormap,
                )
        self.train(is_training)

    def on_before_backward(self, *args, **kwargs) -> None:
        if (self.global_step % self.hparams['viz_frequency']) != 0:
            return
        logger.warning('Starting visualization')
        self._log_sample_images(self.val_dataset, self.viz_val_indices, 'val')
        self._log_sample_images(self.train_dataset, self.viz_train_indices, 'train')

    def on_validation_epoch_end(self) -> None:
        self._log_sample_images(self.val_dataset, self.viz_val_indices, 'val')
        self._log_sample_images(self.train_dataset, self.viz_train_indices, 'train')
