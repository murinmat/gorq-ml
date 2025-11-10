import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
from torch import Tensor

from src.models.utils import MyResample2D
from src.models.HJSuperSloMo import HJSuperSloMo


@dataclass(kw_only=True)
class CycleHJOutput:
    im1: Tensor
    im2: Tensor
    im3: Tensor
    im_t_out: Tensor
    im_target: Tensor
    pred12: Tensor
    pred23: Tensor
    uvb: Tensor
    uvf: Tensor
    pred12_uvb: Tensor
    pred12_uvf: Tensor
    pred23_uvb: Tensor
    pred23_uvf: Tensor
    im12w_raw: Tensor
    im23w_raw: Tensor


class CycleHJSuperSloMo(HJSuperSloMo):
    def __init__(
        self,
        *,
        nc_depth: list[int] = [32, 64, 128, 256, 512],
        flow_scale: float,
        num_interp: int
    ):
        super(CycleHJSuperSloMo, self).__init__(flow_scale=flow_scale, num_interp=num_interp, nc_depth=nc_depth)

    def network_output(self, im1, im2, target_index):
        # Estimate bi-directional optical flows between input low FPS frame pairs
        # Downsample images for robust intermediate flow estimation
        ds_im1 = F.interpolate(im1, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_im2 = F.interpolate(im2, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb = self.make_flow_prediction(torch.cat((ds_im1, ds_im2), dim=1))


        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im1w_raw = self.resample2d(im1, uvb_t_raw)  # im1w_raw
        im2w_raw = self.resample2d(im2, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((im1, im2, im1w_raw, uvb_t_raw, im2w_raw, uvf_t_raw), dim=1)
        uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvb_t = uvb_t_raw + uvb_t # uvb_t
        uvf_t = uvf_t_raw + uvf_t # uvf_t

        im1w = self.resample2d(im1, uvb_t)  # im1w
        im2w = self.resample2d(im2, uvf_t)  # im2w

        # Compute final intermediate frame via weighted blending
        alpha1 = (1 - t) * t_vis_map
        alpha2 = t * (1 - t_vis_map)
        denorm = alpha1 + alpha2 + 1e-10
        im_t_out = (alpha1 * im1w + alpha2 * im2w) / denorm

        return im_t_out, uvb, uvf

    # def network_eval(self, im1, im_target, im2, target_index):
    #     height, width = im1.shape[-2:]
    #     self.resample2d = MyResample2D(width, height).cuda()

    #     im_t_out, uvb, uvf = self.network_output(im1, im2, target_index)

    #     # Calculate losses
    #     losses = {}
    #     losses['pix_loss'] = F.l1_loss(im_t_out, im_target)

    #     # TODO: LPIPS loss?
    #     # im_t_out_features = self.vgg16_features(im_t_out / 255.)
    #     # im_target_features = self.vgg16_features(im_target / 255.)
    #     # losses['vgg16_loss'] = self.L2_loss(im_t_out_features, im_target_features)

    #     losses['warp_loss'] = (
    #         F.l1_loss(self.resample2d(im1, uvb.contiguous()), im2) +
    #         F.l1_loss(self.resample2d(im2, uvf.contiguous()), im1)
    #     )

    #     smooth_bwd = (
    #         F.l1_loss(uvb[:, :, :, :-1], uvb[:, :, :, 1:]) +
    #         F.l1_loss(uvb[:, :, :-1, :], uvb[:, :, 1:, :])
    #     )
    #     smooth_fwd = (
    #         F.l1_loss(uvf[:, :, :, :-1], uvf[:, :, :, 1:]) +
    #         F.l1_loss(uvf[:, :, :-1, :], uvf[:, :, 1:, :])
    #     )

    #     losses['smooth_loss'] = smooth_bwd + smooth_fwd

    #     return losses, im_t_out, im_target

    def forward(self, im1, im2, im3, target_index) -> CycleHJOutput:
        # if not self.training:
        #     return self.network_eval(im1, im2, im3, target_index)
        h, w = im1.shape[-2:]
        self.resample2d = MyResample2D(h, w).to(im1.device)


        # Calculate Pseudo targets at interm_index
        # with torch.no_grad():
        #     _, psuedo_gt12, _ = self.teacher({'image': [im1, im1, im2]}, target_index)
        #     _, psuedo_gt23, _ = self.teacher({'image': [im2, im3, im3]}, target_index)
        # psuedo_gt12, psuedo_gt23 = psuedo_gt12 - self.mean_pix, psuedo_gt23 - self.mean_pix


        pred12, pred12_uvb, pred12_uvf = self.network_output(im1, im2, target_index)
        pred23, pred23_uvb, pred23_uvf = self.network_output(im2, im3, target_index)

        target_index = (self.num_interp + 1) - target_index

        ds_pred12 = F.interpolate(pred12, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_pred23 = F.interpolate(pred23, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb = self.make_flow_prediction(torch.cat((ds_pred12, ds_pred23), dim=1))

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im12w_raw = self.resample2d(pred12, uvb_t_raw)  # im1w_raw
        im23w_raw = self.resample2d(pred23, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((pred12, pred23, im12w_raw, uvb_t_raw, im23w_raw, uvf_t_raw), dim=1)
        uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvb_t = uvb_t_raw + uvb_t # uvb_t
        uvf_t = uvf_t_raw + uvf_t # uvf_t

        im12w = self.resample2d(pred12, uvb_t)  # im1w
        im23w = self.resample2d(pred23, uvf_t)  # im2w

        # Compute final intermediate frame via weighted blending
        alpha1 = (1 - t) * t_vis_map
        alpha2 = t * (1 - t_vis_map)
        denorm = alpha1 + alpha2 + 1e-10
        im_t_out = (alpha1 * im12w + alpha2 * im23w) / denorm
        im_target = im2

        return CycleHJOutput(
            im1=im1,
            im2=im2,
            im3=im3,
            im_t_out=im_t_out,
            im_target=im_target,
            pred12=pred12,
            pred23=pred23,
            uvb=uvb,
            uvf=uvf,
            pred12_uvb=pred12_uvb,
            pred12_uvf=pred12_uvf,
            pred23_uvb=pred23_uvb,
            pred23_uvf=pred23_uvf,
            im12w_raw=im12w_raw,
            im23w_raw=im23w_raw,
        )
