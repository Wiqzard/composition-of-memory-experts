"""
Adapted from Stable Diffusion
https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/autoencoder.py
"""

import torch
import torch.nn as nn

from ..common.modules import (
    Downsample,
    Normalize,
    ResnetBlock2D,
    Upsample,
    make_attn,
    nonlinearity,
)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,  # int or (H, W)
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"

        # ---- NEW: handle (H, W) or int ----
        if isinstance(resolution, int):
            H = W = resolution
        else:
            H, W = resolution
        assert H > 0 and W > 0
        self.resolution = (H, W)

        # sanity: final min side must be divisible by 2^(num_resolutions-1)
        min_side = min(H, W)
        assert (
            min_side % (2 ** (len(ch_mult) - 1)) == 0
        ), f"Min(resolution)={min_side} must be divisible by 2**(len(ch_mult)-1)={2**(len(ch_mult)-1)}"

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_h, curr_w = H, W
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock2D(in_channels=block_in, out_channels=block_out, dropout=dropout)
                )
                block_in = block_out
                # ---- CHANGED: attn condition for non-square ----
                if (curr_h in attn_resolutions) or (curr_w in attn_resolutions):
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, block_in, resamp_with_conv)
                curr_h //= 2
                curr_w //= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,  # int or (H, W)
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"

        # ---- NEW: handle (H, W) or int ----
        if isinstance(resolution, int):
            H = W = resolution
        else:
            H, W = resolution
        assert H > 0 and W > 0
        self.resolution = (H, W)

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # ---- CHANGED: starting latent spatial size per dim ----
        z_h = H // (2 ** (self.num_resolutions - 1))
        z_w = W // (2 ** (self.num_resolutions - 1))
        assert z_h >= 1 and z_w >= 1, "Latent spatial dims must be >= 1"
        self.z_shape = (1, z_channels, z_h, z_w)

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock2D(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        curr_h, curr_w = z_h, z_w
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock2D(in_channels=block_in, out_channels=block_out, dropout=dropout)
                )
                block_in = block_out
                # ---- CHANGED: attn condition for non-square ----
                if (curr_h in attn_resolutions) or (curr_w in attn_resolutions):
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, block_in, resamp_with_conv)
                curr_h *= 2
                curr_w *= 2
            self.up.insert(0, up)  # prepend to maintain order

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # optional: check channel count only, spatial can vary as long as it matches the pyramid
        # assert z.shape[1] == self.z_shape[1], "Latent channels mismatch"

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


# class Encoder(nn.Module):
#    def __init__(
#        self,
#        *,
#        ch,
#        out_ch,
#        ch_mult=(1, 2, 4, 8),
#        num_res_blocks,
#        attn_resolutions,
#        dropout=0.0,
#        resamp_with_conv=True,
#        in_channels,
#        resolution,
#        z_channels,
#        double_z=True,
#        use_linear_attn=False,
#        attn_type="vanilla",
#        **ignore_kwargs,
#    ):
#        super().__init__()
#        if use_linear_attn:
#            attn_type = "linear"
#        self.ch = ch
#        self.num_resolutions = len(ch_mult)
#        self.num_res_blocks = num_res_blocks
#        self.resolution = resolution
#        self.in_channels = in_channels
#
#        # downsampling
#        self.conv_in = torch.nn.Conv2d(
#            in_channels, self.ch, kernel_size=3, stride=1, padding=1
#        )
#
#        curr_res = resolution
#        in_ch_mult = (1,) + tuple(ch_mult)
#        self.in_ch_mult = in_ch_mult
#        self.down = nn.ModuleList()
#        for i_level in range(self.num_resolutions):
#            block = nn.ModuleList()
#            attn = nn.ModuleList()
#            block_in = ch * in_ch_mult[i_level]
#            block_out = ch * ch_mult[i_level]
#            for i_block in range(self.num_res_blocks):
#                block.append(
#                    ResnetBlock2D(
#                        in_channels=block_in,
#                        out_channels=block_out,
#                        dropout=dropout,
#                    )
#                )
#                block_in = block_out
#                if curr_res in attn_resolutions:
#                    attn.append(make_attn(block_in, attn_type=attn_type))
#            down = nn.Module()
#            down.block = block
#            down.attn = attn
#            if i_level != self.num_resolutions - 1:
#                down.downsample = Downsample(block_in, block_in, resamp_with_conv)
#                curr_res = curr_res // 2
#            self.down.append(down)
#
#        # middle
#        self.mid = nn.Module()
#        self.mid.block_1 = ResnetBlock2D(
#            in_channels=block_in,
#            out_channels=block_in,
#            dropout=dropout,
#        )
#        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
#        self.mid.block_2 = ResnetBlock2D(
#            in_channels=block_in,
#            out_channels=block_in,
#            dropout=dropout,
#        )
#
#        # end
#        self.norm_out = Normalize(block_in)
#        self.conv_out = torch.nn.Conv2d(
#            block_in,
#            2 * z_channels if double_z else z_channels,
#            kernel_size=3,
#            stride=1,
#            padding=1,
#        )
#
#    def forward(self, x):
#        # downsampling
#        hs = [self.conv_in(x)]
#        for i_level in range(self.num_resolutions):
#            for i_block in range(self.num_res_blocks):
#                h = self.down[i_level].block[i_block](hs[-1])
#                if len(self.down[i_level].attn) > 0:
#                    h = self.down[i_level].attn[i_block](h)
#                hs.append(h)
#            if i_level != self.num_resolutions - 1:
#                hs.append(self.down[i_level].downsample(hs[-1]))
#
#        # middle
#        h = hs[-1]
#        h = self.mid.block_1(h)
#        h = self.mid.attn_1(h)
#        h = self.mid.block_2(h)
#
#        # end
#        h = self.norm_out(h)
#        h = nonlinearity(h)
#        h = self.conv_out(h)
#        return h
#
#
# class Decoder(nn.Module):
#    def __init__(
#        self,
#        *,
#        ch,
#        out_ch,
#        ch_mult=(1, 2, 4, 8),
#        num_res_blocks,
#        attn_resolutions,
#        dropout=0.0,
#        resamp_with_conv=True,
#        in_channels,
#        resolution,
#        z_channels,
#        give_pre_end=False,
#        tanh_out=False,
#        use_linear_attn=False,
#        attn_type="vanilla",
#        **ignorekwargs,
#    ):
#        super().__init__()
#        if use_linear_attn:
#            attn_type = "linear"
#        self.ch = ch
#        self.num_resolutions = len(ch_mult)
#        self.num_res_blocks = num_res_blocks
#        self.resolution = resolution
#        self.in_channels = in_channels
#        self.give_pre_end = give_pre_end
#        self.tanh_out = tanh_out
#
#        # compute in_ch_mult, block_in and curr_res at lowest res
#        in_ch_mult = (1,) + tuple(ch_mult)
#        block_in = ch * ch_mult[self.num_resolutions - 1]
#        curr_res = resolution // 2 ** (self.num_resolutions - 1)
#        self.z_shape = (1, z_channels, curr_res, curr_res)
#
#        # z to block_in
#        self.conv_in = torch.nn.Conv2d(
#            z_channels, block_in, kernel_size=3, stride=1, padding=1
#        )
#
#        # middle
#        self.mid = nn.Module()
#        self.mid.block_1 = ResnetBlock2D(
#            in_channels=block_in,
#            out_channels=block_in,
#            dropout=dropout,
#        )
#        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
#        self.mid.block_2 = ResnetBlock2D(
#            in_channels=block_in,
#            out_channels=block_in,
#            dropout=dropout,
#        )
#
#        # upsampling
#        self.up = nn.ModuleList()
#        for i_level in reversed(range(self.num_resolutions)):
#            block = nn.ModuleList()
#            attn = nn.ModuleList()
#            block_out = ch * ch_mult[i_level]
#            for i_block in range(self.num_res_blocks + 1):
#                block.append(
#                    ResnetBlock2D(
#                        in_channels=block_in,
#                        out_channels=block_out,
#                        dropout=dropout,
#                    )
#                )
#                block_in = block_out
#                if curr_res in attn_resolutions:
#                    attn.append(make_attn(block_in, attn_type=attn_type))
#            up = nn.Module()
#            up.block = block
#            up.attn = attn
#            if i_level != 0:
#                up.upsample = Upsample(block_in, block_in, resamp_with_conv)
#                curr_res = curr_res * 2
#            self.up.insert(0, up)  # prepend to get consistent order
#
#        # end
#        self.norm_out = Normalize(block_in)
#        self.conv_out = torch.nn.Conv2d(
#            block_in, out_ch, kernel_size=3, stride=1, padding=1
#        )
#
#    def forward(self, z):
#        # assert z.shape[1:] == self.z_shape[1:]
#        self.last_z_shape = z.shape
#
#        # z to block_in
#        h = self.conv_in(z)
#
#        # middle
#        h = self.mid.block_1(h)
#        h = self.mid.attn_1(h)
#        h = self.mid.block_2(h)
#
#        # upsampling
#        for i_level in reversed(range(self.num_resolutions)):
#            for i_block in range(self.num_res_blocks + 1):
#                h = self.up[i_level].block[i_block](h)
#                if len(self.up[i_level].attn) > 0:
#                    h = self.up[i_level].attn[i_block](h)
#            if i_level != 0:
#                h = self.up[i_level].upsample(h)
#
#        # end
#        if self.give_pre_end:
#            return h
#
#        h = self.norm_out(h)
#        h = nonlinearity(h)
#        h = self.conv_out(h)
#        if self.tanh_out:
#            h = torch.tanh(h)
#        return h
#
