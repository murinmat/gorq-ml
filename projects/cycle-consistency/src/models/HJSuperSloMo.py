import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
import lightning as L
import torch.nn.functional as F

from src.models.utils import MyResample2D


class BaseConvLayer(nn.Module):
    def __init__(self, nc_out: int, upsample: bool, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity(),
            nn.LazyConv2d(nc_out, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(nc_out, nc_out, kernel_size, padding=padding),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SkipConnLayer(nn.Module):
    def __init__(self, module: nn.Module, concat_dim: int):
        super().__init__()
        self.module = module
        self.concat_dim = concat_dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x, self.module(x)], dim=self.concat_dim)


class HJSuperSloMo(nn.Module):
    def __init__(
        self,
        *,
        nc_depth: list[int] = [32, 64, 128, 256, 512],
        flow_scale: float,
        num_interp: int
    ):
        super(HJSuperSloMo, self).__init__()
        self.is_output_flow = False
        self.num_interp = num_interp
        self.scale = flow_scale

        # --------------------- encoder --------------------
        idx_kernel_sizes = {0: 7, 1: 5}
        self.flow_pred_encoder_layers = nn.ModuleList()
        self.flow_pred_decoder_layers = nn.ModuleList()
        self.flow_interp_encoder_layers = nn.ModuleList()
        self.flow_interp_decoder_layers = nn.ModuleList()
        for idx, nc_out in enumerate(nc_depth):
            kernel_size = idx_kernel_sizes.get(idx, 3)
            # Encoders
            self.flow_pred_encoder_layers.append(
                BaseConvLayer(nc_out, False, kernel_size, kernel_size//2)
            )
            self.flow_interp_encoder_layers.append(
                BaseConvLayer(nc_out, False, kernel_size, kernel_size//2)
            )
            # Decoders
            self.flow_pred_decoder_layers.append(
                BaseConvLayer(nc_out, True)
            )
            self.flow_interp_decoder_layers.append(
                BaseConvLayer(nc_out, True)
            )

        self.flow_pred_bottleneck = BaseConvLayer(nc_depth[-1], False)
        self.flow_pred_refine_layer = nn.Sequential(
            nn.Conv2d(nc_depth[0]*2, nc_depth[0], 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )
        self.forward_flow_conv = nn.Conv2d(nc_depth[0], 2, 1)
        self.backward_flow_conv = nn.Conv2d(nc_depth[0], 2, 1)

        self.flow_interp_bottleneck = BaseConvLayer(nc_depth[-1], False)

        self.flow_interp_refine_layer = nn.Sequential(
            nn.Conv2d(nc_depth[0]*2, nc_depth[0], 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

        self.flow_interp_forward_out_layer = nn.Conv2d(nc_depth[0], 2, 1)
        self.flow_interp_backward_out_layer = nn.Conv2d(nc_depth[0], 2, 1)

        # visibility
        self.flow_interp_vis_layer = nn.Conv2d(nc_depth[0], 1, 1)

        self.ignore_keys = ['vgg', 'grid_w', 'grid_h', 'tlinespace', 'resample2d']
        self.register_buffer('tlinespace', torch.linspace(0, 1, 2 + num_interp).float())

        # vgg16 = torchvision.models.vgg16(pretrained=True)
        # self.vgg16_features = nn.Sequential(*list(vgg16.children())[0][:22])
        # for param in self.vgg16_features.parameters():
        #     param.requires_grad = False

        # loss weights
        # self.pix_alpha = 0.8
        # self.warp_alpha = 0.4
        # self.vgg16_alpha = 0.005
        # self.smooth_alpha = 1.

    def make_flow_interpolation(self, x: Tensor, flow_pred_bottleneck_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Encoder
        encoder_out = []
        for module in self.flow_interp_encoder_layers:
            out = module(x)
            encoder_out.append(out)
            x = F.avg_pool2d(out, kernel_size=2, stride=2)
        # Bottleneck
        x = torch.cat([flow_pred_bottleneck_out, self.flow_interp_bottleneck(x)], dim=1)
        # Decoder
        for module, out_skip_conn in zip(reversed(self.flow_interp_decoder_layers), reversed(encoder_out)):
            x = torch.cat([module(x), out_skip_conn], dim=-3)

        # Get all final outputs
        flow_interp_motion_rep = self.flow_interp_refine_layer(x)
        flow_interp_forward_flow = self.flow_interp_forward_out_layer(flow_interp_motion_rep)
        flow_interp_backward_flow = self.flow_interp_backward_out_layer(flow_interp_motion_rep)
        flow_interp_vis_map = self.flow_interp_vis_layer(flow_interp_motion_rep)
        flow_interp_vis_map = torch.sigmoid(flow_interp_vis_map)

        return flow_interp_forward_flow, flow_interp_backward_flow, flow_interp_vis_map

    def make_flow_prediction(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Encoder
        encoder_out = []
        for module in self.flow_pred_encoder_layers:
            out = module(x)
            encoder_out.append(out)
            x = F.avg_pool2d(out, kernel_size=2, stride=2)
        # Bottleneck
        bottleneck_out = self.flow_pred_bottleneck(x)
        x = bottleneck_out
        # Decoder
        for module, out_skip_conn in zip(reversed(self.flow_pred_decoder_layers), reversed(encoder_out)):
            x = torch.cat([module(x), out_skip_conn], dim=-3)

        # Get all final outputs
        motion_rep = self.flow_pred_refine_layer(x)
        uvf = self.forward_flow_conv(motion_rep)
        uvb = self.backward_flow_conv(motion_rep)

        return uvf, bottleneck_out, uvb
