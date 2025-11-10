import numpy as np
import torch
import torch.nn as nn


class MyResample2D(nn.Module):
    def __init__(self, width, height):
        super(MyResample2D, self).__init__()

        self.width = width
        self.height = height

        # make grids for horizontal and vertical displacements
        grid_w, grid_h = np.meshgrid(np.arange(width), np.arange(height))
        grid_w, grid_h = grid_w.reshape((1,) + grid_w.shape), grid_h.reshape((1,) + grid_h.shape)

        self.register_buffer("grid_w", torch.as_tensor(grid_w, dtype=torch.float32))
        self.register_buffer("grid_h", torch.as_tensor(grid_h, dtype=torch.float32))

    def forward(self, im, uv):

        # Get relative displacement
        u = uv[:, 0, ...]
        v = uv[:, 1, ...]

        # Calculate absolute displacement along height and width axis -> (batch_size, height, width)
        ww = self.grid_w.expand_as(u) + u
        hh = self.grid_h.expand_as(v) + v

        # Normalize indices to [-1,1]
        ww = 2 * ww / (self.width - 1) - 1
        hh = 2 * hh / (self.height - 1) - 1

        # Form a grid of shape (batch_size, height, width, 2)
        norm_grid_wh = torch.stack((ww, hh), dim=-1)

        # Perform a resample
        reampled_im = torch.nn.functional.grid_sample(im, norm_grid_wh, align_corners=True)

        return reampled_im
