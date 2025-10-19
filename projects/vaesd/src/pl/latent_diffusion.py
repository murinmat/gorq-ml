import torch
import torch.nn.functional as F
import lightning as L
from tqdm.auto import tqdm
from loguru import logger
from torch.utils.data import Dataset
from typing import Literal, Tuple
from torch.optim import Optimizer
from torch.optim.adam import Adam


from src.models.vqvae import VQVAE
from src.pl.vae import VAELightning
from src.models.unet_base import Unet
from src.noise_scheduler import LinearNoiseScheduler

from gorq_ml.training.utils import log_image


class LatentDiffusionLightning(L.LightningModule):
    def __init__(
            self,
            model_kwargs: dict,
            model_lr: float,
            model_betas: Tuple[float, float],
            vae_model_name: str,
            vae_ckpt_path: str,
            noise_scheduler_kwargs: dict,
            viz_frequency: int,
            viz_n_samples: int,
            viz_denoising_seed: int,
            input_img_resolution: int,
            num_inference_tsteps: int,
            latent_shape: list[int],
            plot_colormap: str | None = None,
            plot_per_channel: bool = False,
    ) -> None:
        super().__init__()
        self.model = Unet(**model_kwargs)
        self.save_hyperparameters(ignore=['vae'])
        # TODO: Load the VAE
        self.vae: VQVAE = globals()[vae_model_name].load_from_checkpoint(vae_ckpt_path).model.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
        self.noise_scheduler = LinearNoiseScheduler(**noise_scheduler_kwargs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(
            self.model.parameters(),
            lr=self.hparams['model_lr'],
            betas=self.hparams['model_betas'],
        )
    
    def calc_loss(self, batch: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.vae.encode(batch)[0]
        noise = torch.randn_like(encoded)
        t = torch.randint(
            0,
            self.noise_scheduler.num_timesteps,
            (encoded.shape[0],),
            device=encoded.device,
        )
        noisy_encoded = self.noise_scheduler.add_noise(encoded, noise, t)
        noise_pred = self.model(noisy_encoded, t)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        loss = self.calc_loss(batch)
        self.log('loss/train', loss.item(), on_epoch=True, on_step=True, prog_bar=True)

        return loss
        
    def validation_step(self, batch: torch.Tensor) -> None:
        loss = self.calc_loss(batch)
        self.log('loss/val', loss.item(), on_epoch=True, on_step=False)

    @torch.inference_mode()
    @torch.autocast('cuda')
    def sample(self, num_samples: int, seed: int, device: torch.device, eta: float = 0.0) -> torch.Tensor:
        xt = torch.randn(
            (
                num_samples,
                *self.hparams['latent_shape']
            ),
            generator=torch.Generator().manual_seed(seed)
        ).to(self.device)

        t_start = self.noise_scheduler.num_timesteps - 1
        t_end = 0
        ddim_timesteps = torch.linspace(t_start, t_end, self.hparams['num_inference_tsteps']).long()
        alphas = self.noise_scheduler.alpha_cum_prod.to(device)


        for i, current_t in tqdm(enumerate(ddim_timesteps), total=len(ddim_timesteps), desc='Sampling in progress', leave=False):
            noise_pred = self.model(xt, torch.as_tensor(current_t).unsqueeze(0).to(device))
            
            alpha_t = alphas[current_t]
            x0_pred = (xt - (1 - alpha_t).sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)

            if i < len(ddim_timesteps) - 1:
                next_t = ddim_timesteps[i + 1]
                alpha_next = alphas[next_t]
                
                # DDIM update formula
                xt = alpha_next.sqrt() * x0_pred + ((1 - alpha_next) - eta**2 * (1 - alpha_t) / (1 - alpha_next)).sqrt() * noise_pred
            else:
                xt = x0_pred  # final step

        quantized_output = self.vae.quantize(xt)[0]
        return self.vae.decode(quantized_output).cpu().clamp(-1, 1) * 0.5 + 0.5

    def _viz_samples(self):
        logger.warning(f'Starting visualization')
        sampled_outputs = self.sample(
            num_samples=self.hparams['viz_n_samples'],
            seed=self.hparams['viz_denoising_seed'],
            device=self.device,
        )
        
        colormap = self.hparams.get('plot_colormap', None)
        for idx, output in enumerate(sampled_outputs):
            if self.hparams['plot_per_channel']:
                for c_idx, c_out in enumerate(output):
                    log_image(
                        data=c_out,
                        title=f'Samples ({c_idx})',
                        series=f'Index {idx}',
                        iteration=self.global_step,
                        max_history=-1,
                        colormap=colormap,
                    )
            else:
                log_image(
                    data=output[0] if colormap is not None else output.permute(1, 2, 0),
                    title=f'Samples',
                    series=f'Index {idx}',
                    iteration=self.global_step,
                    max_history=-1,
                    colormap=colormap,
                )
        logger.warning(f'Finished visualization')

    def on_after_backward(self, *args, **kwargs) -> None:
        if ((self.global_step - 1) % self.hparams['viz_frequency']) != 0:
            return
        self._viz_samples()

    def on_validation_epoch_end(self) -> None:
        self._viz_samples()
