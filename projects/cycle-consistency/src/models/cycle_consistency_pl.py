import lightning as L
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import lightning as L
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from torch.optim import Optimizer
from torch.optim.adamw import AdamW
from clearml.logger import Logger
from torch import Tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LRScheduler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from gorq_ml.training.utils import log_image

from src.models.discriminator import Discriminator
from src.models.CycleHJSuperSloMo import CycleHJSuperSloMo, CycleHJOutput

class CycleConsistencyPL(L.LightningModule):
    def __init__(
        self,
        model_kwargs: dict,
        discriminator_kwargs: dict,
        model_lr: float,
        model_betas: Tuple[float, float],
        disc_lr: float,
        disc_betas: Tuple[float, float],
        disc_step_start: int,
        pixel_weight: float,
        disc_weight: float,
        warp_loss_weight: float,
        smooth_loss_weight: float,
        viz_val_indices: list[int],
        viz_train_indices: list[int],
        viz_frequency: int,
        perceptual_weight: float = 0.,
        val_dataloader: DataLoader | None = None,
        train_dataloader: DataLoader | None = None,
        plot_colormap: str | None = None,
        plot_per_channel: bool = False,
        lr_warmup_steps: int = 1,
        num_training_steps: int | None = None,
        acc_grad: int = 1,
    ):
        super().__init__()
        self.lpips = LPIPS(net_type='vgg').eval()
        self.model = CycleHJSuperSloMo(**model_kwargs)
        self.discriminator = Discriminator(**discriminator_kwargs)
        self.automatic_optimization = False
        self.viz_val_indices = viz_val_indices
        self.viz_train_indices = viz_train_indices
        self.train_codebook_counts = 0.
        self.val_codebook_counts = 0.
        self._val_dataloader = val_dataloader
        self._train_dataloader = train_dataloader
        self._last_viz_tstep = -1
        self.num_steps_taken = 0
        self._last_viz_tstep = -1
        self.save_hyperparameters(ignore=['lpips', 'train_dataloader', 'val_dataloader', '_last_viz_tstep'])

    def _should_discriminator_be_trained(self) -> bool:
        return self.num_steps_taken > self.hparams['disc_step_start']
    
    def configure_optimizers(self) -> Tuple[list[Optimizer], list[LRScheduler]]:
        optim_ae = AdamW(
            self.model.parameters(),
            lr=self.hparams['model_lr'],
            betas=self.hparams['model_betas'],
        )
        optim_disc = AdamW(
            self.discriminator.parameters(),
            lr=self.hparams['disc_lr'],
            betas=self.hparams['disc_betas']
        )
        schedulers_ae: list[LRScheduler] = [
            LinearLR(
                optimizer=optim_ae,
                start_factor=0.001,
                end_factor=1,
                total_iters=self.hparams['lr_warmup_steps'],
            )
        ]
        schedulers_disc: list[LRScheduler] = [
            LinearLR(
                optimizer=optim_disc,
                start_factor=0.001,
                end_factor=1,
                total_iters=self.hparams['lr_warmup_steps'],
            )
        ]
        if self.hparams['num_training_steps'] is not None and self._train_dataloader is not None:
            schedulers_ae.append(
                CosineAnnealingLR(
                    optimizer=optim_ae,
                    T_max=self.hparams['num_training_steps'],
                )
            )
            schedulers_disc.append(
                CosineAnnealingLR(
                    optimizer=optim_disc,
                    T_max=self.hparams['num_training_steps'],
                )
            )
        schedulers_ae = SequentialLR(
            optimizer=optim_ae,
            schedulers=schedulers_ae,
            milestones=[self.hparams['lr_warmup_steps']] if len(schedulers_ae) == 2 else []
        )
        schedulers_disc = SequentialLR(
            optimizer=optim_disc,
            schedulers=schedulers_disc,
            milestones=[self.hparams['lr_warmup_steps']] if len(schedulers_disc) == 2 else []
        )

        return ([optim_ae, optim_disc], [schedulers_ae, schedulers_disc]) # type: ignore

    def get_discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        disc_fake_pred = self.discriminator(fake.detach())
        disc_real_pred = self.discriminator(real)
        disc_fake_loss = self.hparams['disc_weight'] * F.mse_loss(
            disc_fake_pred, torch.zeros_like(disc_fake_pred)
        )
        disc_real_loss = self.hparams['disc_weight'] * F.mse_loss(
            disc_real_pred, torch.ones_like(disc_real_pred)
        )
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        return disc_loss, {
            'disc_fake_loss': disc_fake_loss.item(),
            'disc_real_loss': disc_real_loss.item(),
        }

    def get_generator_loss(self, out: CycleHJOutput) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_dict = {}
        final_loss = 0

        # Pixel loss
        loss_dict['pixel_loss'] = F.l1_loss(out.im_t_out, out.im_target) * self.hparams['pixel_weight']
        final_loss += loss_dict['pixel_loss']

        # Warp loss
        loss_dict['warp_loss'] = (
            F.l1_loss(out.im12w_raw, out.im2) +
            F.l1_loss(out.im23w_raw, out.im2) +
            F.l1_loss(self.model.resample2d(out.pred12, out.uvb), out.pred23) +
            F.l1_loss(self.model.resample2d(out.pred23, out.uvf), out.pred12) +
            F.l1_loss(self.model.resample2d(out.im1, out.pred12_uvb), out.im2) +
            F.l1_loss(self.model.resample2d(out.im2, out.pred12_uvf), out.im1) +
            F.l1_loss(self.model.resample2d(out.im2, out.pred23_uvb), out.im3) +
            F.l1_loss(self.model.resample2d(out.im3, out.pred23_uvf), out.im2)
        )
        final_loss += loss_dict['warp_loss'] * self.hparams['warp_loss_weight']

        # Smooth loss
        smooth_bwd = (
            F.l1_loss(out.uvb[:, :, :, :-1], out.uvb[:, :, :, 1:]) +
            F.l1_loss(out.uvb[:, :, :-1, :], out.uvb[:, :, 1:, :]) +
            F.l1_loss(out.pred12_uvb[:, :, :, :-1], out.pred12_uvb[:, :, :, 1:]) +
            F.l1_loss(out.pred12_uvb[:, :, :-1, :], out.pred12_uvb[:, :, 1:, :]) +
            F.l1_loss(out.pred23_uvb[:, :, :, :-1], out.pred23_uvb[:, :, :, 1:]) +
            F.l1_loss(out.pred23_uvb[:, :, :-1, :], out.pred23_uvb[:, :, 1:, :])
        )
        smooth_fwd = (
            F.l1_loss(out.uvf[:, :, :, :-1], out.uvf[:, :, :, 1:]) +
            F.l1_loss(out.uvf[:, :, :-1, :], out.uvf[:, :, 1:, :]) +
            F.l1_loss(out.pred12_uvf[:, :, :, :-1], out.pred12_uvf[:, :, :, 1:]) +
            F.l1_loss(out.pred12_uvf[:, :, :-1, :], out.pred12_uvf[:, :, 1:, :]) +
            F.l1_loss(out.pred23_uvf[:, :, :, :-1], out.pred23_uvf[:, :, :, 1:]) +
            F.l1_loss(out.pred23_uvf[:, :, :-1, :], out.pred23_uvf[:, :, 1:, :])
        )
        loss_dict['smooth_loss'] = (smooth_bwd + smooth_fwd) * self.hparams['smooth_loss_weight']
        final_loss += loss_dict['smooth_loss']

        # Perceptual loss (LPIPS)
        if self.hparams['perceptual_weight'] > 0:
            out_for_lpips = (out.im_t_out * 2 - 1).clamp(-1, 1).float()
            inp_for_lpips = (out.im_target * 2 - 1).clamp(-1, 1).float()
            if out_for_lpips.shape[-3] == 1:
                out_for_lpips = torch.cat([out_for_lpips] * 3, dim=-3)
                inp_for_lpips = torch.cat([inp_for_lpips] * 3, dim=-3)
            perceptual_loss = self.hparams['perceptual_weight'] * self.lpips(
                out_for_lpips, inp_for_lpips
            ).mean()
            loss_dict['perceptual_loss'] = perceptual_loss.item()
            final_loss += perceptual_loss
        # Discriminator only if the training of it should have started by now
        if self._should_discriminator_be_trained():
            disc_input = torch.cat([
                torch.cat([out.im1, out.pred12, out.im2], dim=-3),
                torch.cat([out.im2, out.pred23, out.im3], dim=-3),
            ])
            disc_fake_pred = self.discriminator(disc_input)
            gen_disc_loss = self.hparams['disc_weight'] * F.mse_loss(
                disc_fake_pred, torch.ones_like(disc_fake_pred)
            )
            loss_dict['gen_disc_loss'] = gen_disc_loss.item()
            final_loss += gen_disc_loss
        else:
            loss_dict['gen_disc_loss'] = 1
            final_loss += 1
        return final_loss, loss_dict

    def train_generator(self, out: CycleHJOutput, batch_idx: int) -> None:
        optim: Optimizer = self.optimizers()[0]  # type: ignore
        lr_scheduler: LRScheduler = self.lr_schedulers()[0] # type: ignore

        self.toggle_optimizer(optim)
        loss, loss_dict = self.get_generator_loss(out)
        loss /= self.hparams['acc_grad']
        self.manual_backward(loss)
        if (batch_idx + 1) % self.hparams['acc_grad'] == 0:
            optim.step()
            optim.zero_grad()
            lr_scheduler.step()
            self.num_steps_taken += 1
        self.untoggle_optimizer(optim)
        self.log('gen_loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'{k}/train': v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)

    def get_discriminator_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        disc_fake_pred = self.discriminator(fake.detach())
        disc_real_pred = self.discriminator(real)
        disc_fake_loss = self.hparams['disc_weight'] * F.mse_loss(
            disc_fake_pred, torch.zeros_like(disc_fake_pred)
        )
        disc_real_loss = self.hparams['disc_weight'] * F.mse_loss(
            disc_real_pred, torch.ones_like(disc_real_pred)
        )
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        return disc_loss, {
            'disc_fake_loss': disc_fake_loss.item(),
            'disc_real_loss': disc_real_loss.item(),
        }

    def train_discriminator(
        self,
        out: CycleHJOutput,
        batch_idx: int,
    ) -> None:
        optim: Optimizer = self.optimizers()[1]  # type: ignore
        lr_scheduler: LRScheduler = self.lr_schedulers()[1] # type: ignore

        self.toggle_optimizer(optim)
        loss, loss_dict = self.get_discriminator_loss(
            real=torch.cat([out.im1, out.im2, out.im3], dim=-3),
            fake=torch.cat(
                [out.im1, out.pred12, out.im2] if torch.randn((1,)) > .5 else [out.im2, out.pred23, out.im3],
                dim=-3,
            ),
        )
        loss /= self.hparams['acc_grad']
        self.manual_backward(loss)
        if (batch_idx + 1) % self.hparams['acc_grad'] == 0:
            optim.step()
            optim.zero_grad()
            lr_scheduler.step()
        self.untoggle_optimizer(optim)

        self.log(f'disc_loss/train', loss.item(), on_step=True, on_epoch=True)
        self.log_dict({f'{k}/train': v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model.forward(batch['im1'], batch['im2'], batch['im3'], batch['t'])
        self.train_generator(out, batch_idx)
        if self._should_discriminator_be_trained():
            self.train_discriminator(out, batch_idx)

    
    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        out = self.model.forward(batch['im1'], batch['im2'], batch['im3'], batch['t'])
        gen_loss, gen_loss_dict = self.get_generator_loss(out)
        disc_loss, disc_loss_dict = self.get_discriminator_loss(
            real=torch.cat([out.im1, out.im2, out.im3], dim=-3),
            fake=torch.cat(
                [out.im1, out.pred12, out.im2] if torch.randn((1,)) > .5 else [out.im2, out.pred23, out.im3],
                dim=-3,
            ),
        )
        self.log('gen_loss/val', gen_loss.item(), on_step=False, on_epoch=True)
        self.log(f'disc_loss/val', disc_loss.item(), on_step=False, on_epoch=True)
        self.log_dict(
            {
                f'{k}/val': v
                for k, v in (gen_loss_dict | disc_loss_dict).items()
            },
            on_step=False,
            on_epoch=True
        )


    @torch.no_grad()
    def _log_sample_images(self, ds: Dataset | None, indices: list[int], type: str):
        is_training = self.training
        self.train(False)
        if ds is None:
            return
        colormap = self.hparams.get('plot_colormap', 'Greys')
        all_interp = self.model.num_interp
        for idx, val_sample_idx in enumerate(indices):
            sample = ds[val_sample_idx]
            batch = {
                'im1': sample['im1'].to(self.device)[None],
                'im2': sample['im3'].to(self.device)[None],
                'im3': sample['im3'].to(self.device)[None],
                't': sample['t'].to(self.device)[None]
            }
            out = self.model.forward(batch['im1'], batch['im2'], batch['im3'], batch['t'])
            im_tar_recon = torch.cat(
                [
                    out.im_t_out,
                    torch.ones_like(out.im_target[..., :2]),
                    out.im_target,
                ],
                dim=-1
            )[0, 0, :, :].cpu()
            log_image(
                data=im_tar_recon,
                title=f'Viz {type}',
                series=f'Index {idx} (t={sample["t"]})',
                iteration=self.global_step,
                max_history=-1,
                colormap=colormap,
            )
            interp_outs = []
            for t in range(all_interp):
                batch['t'] = torch.as_tensor([t], device=self.device)
                out = self.model.forward(batch['im1'], batch['im2'], batch['im3'], batch['t'])
                interp_outs.append(out.pred12)
                interp_outs.append(torch.ones_like(out.im_target[..., :2]))
            im_all_interp = torch.cat((
                out.im1,
                torch.ones_like(out.im_target[..., :2]),
                *interp_outs,
                out.im2,
            ), dim=-1)[0, 0, :, :].cpu()
            log_image(
                data=im_all_interp,
                title=f'Viz {type} (full interp)',
                series=f'Index {idx}',
                iteration=self.global_step,
                max_history=-1,
                colormap=colormap,
            )

        self.train(is_training)

    def on_validation_epoch_end(self) -> None:
        self._log_sample_images(self._val_dataloader.dataset, self.viz_val_indices, 'val')
        self._log_sample_images(self._train_dataloader.dataset, self.viz_train_indices, 'train')

    def on_before_backward(self, *args, **kwargs) -> None:
        if (self.global_step % self.hparams['viz_frequency']) != 0:
            return
        if self._last_viz_tstep == self.global_step:
            return
        self._last_viz_tstep = self.global_step
        logger.warning('Starting visualization')
        self._log_sample_images(self._val_dataloader.dataset, self.viz_val_indices, 'val')
        self._log_sample_images(self._train_dataloader.dataset, self.viz_train_indices, 'train')
        logger.warning('Ended visualization')