import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.proj(emb)


class Conditioner(nn.Module):
    """Multi-scale condition encoder with progressive downsampling."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels * 2, 5, stride=2, padding=2),
            nn.GroupNorm(8, out_channels * 2),
            nn.SiLU())
        self.downsample2 = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels * 4, 5, stride=2, padding=2),
            nn.GroupNorm(8, out_channels * 4),
            nn.SiLU())
        self.downsample3 = nn.Sequential(
            nn.Conv3d(out_channels * 4, out_channels * 8, 5, stride=2, padding=2),
            nn.GroupNorm(8, out_channels * 8),
            nn.SiLU())
        self.downsample4 = nn.Sequential(
            nn.Conv3d(out_channels * 8, out_channels * 16, 5, stride=2, padding=2),
            nn.GroupNorm(8, out_channels * 16),
            nn.SiLU())

    def forward(self, cond):
        B, C, D, H, W, _ = cond.shape
        cond = cond.permute(0, 5, 1, 2, 3, 4).contiguous()
        cond = cond.view(B, -1, D, H, W)
        cond0 = self.conv(cond)
        cond1 = self.downsample1(cond0)
        cond2 = self.downsample2(cond1)
        cond3 = self.downsample3(cond2)
        cond4 = self.downsample4(cond3)
        return [cond0, cond1, cond2, cond3, cond4]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim, cond_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.t_proj = nn.Linear(t_dim, out_channels * 2)
        self.cond_proj = nn.Conv3d(cond_channels, out_channels * 2, 1)

    def forward(self, x, t_emb, cond):
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.act(h)

        t_emb = self.t_proj(t_emb)
        t_scale, t_shift = torch.chunk(t_emb, 2, dim=1)
        h = h * (1 + t_scale[:, :, None, None, None]) + t_shift[:, :, None, None, None]

        if cond is not None:
            cond = self.cond_proj(cond)
            cond_scale, cond_shift = torch.chunk(cond, 2, dim=1)
            h = h * (1 + cond_scale) + cond_shift

        h = self.norm2(h)
        h = self.conv2(h)
        h = self.act(h)

        return h + self.shortcut(x)


class Attention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).chunk(3, dim=1)

        q, k, v = map(lambda t: t.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C).contiguous(), qkv)

        attn = torch.bmm(q, k.permute(0, 2, 1)) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        h = torch.bmm(attn, v).reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet4D(nn.Module):
    """4D UNet for voxel-based motion diffusion.

    Input:  x    - (B, T, J, D, H, W) noised voxel probability grids
            cond - (B, T, D, H, W, 3) condition voxel grids
            text_features - (B, 512) CLIP text features
            t    - (B,) diffusion timesteps

    Output: (B, T, J, D, H, W) predicted clean voxel probability grids
    """
    def __init__(self, in_channels=40, cond_channels=3, t_dim=128):
        super().__init__()
        self.t_embed = TimestepEmbedder(t_dim)
        self.base_channel = 128

        self.input = in_channels * 22
        self.cond_out_channel = 128

        self.text_proj = nn.Linear(512, t_dim)

        self.cond_proj = Conditioner(in_channels, self.cond_out_channel)

        self.enc1 = nn.ModuleList([
            ResidualBlock(self.input, self.base_channel, t_dim, self.cond_out_channel),
            Downsample(self.base_channel)
        ])
        self.enc2 = nn.ModuleList([
            ResidualBlock(self.base_channel, self.base_channel * 2, t_dim, self.cond_out_channel * 2),
            Attention3D(self.base_channel * 2),
            Downsample(self.base_channel * 2)
        ])
        self.enc3 = nn.ModuleList([
            ResidualBlock(self.base_channel * 2, self.base_channel * 4, t_dim, self.cond_out_channel * 4),
            Attention3D(self.base_channel * 4),
            Downsample(self.base_channel * 4)
        ])
        self.enc4 = nn.ModuleList([
            ResidualBlock(self.base_channel * 4, self.base_channel * 8, t_dim, self.cond_out_channel * 8),
            Attention3D(self.base_channel * 8),
            Downsample(self.base_channel * 8)
        ])

        self.bneck = nn.ModuleList([
            ResidualBlock(self.base_channel * 8, self.base_channel * 8, t_dim, self.cond_out_channel * 16),
            Attention3D(self.base_channel * 8),
            ResidualBlock(self.base_channel * 8, self.base_channel * 8, t_dim, self.cond_out_channel * 16)
        ])

        self.dec4 = nn.ModuleList([
            ResidualBlock(self.base_channel * 16, self.base_channel * 8, t_dim, self.cond_out_channel * 16),
            Attention3D(self.base_channel * 8),
            Upsample(self.base_channel * 8)
        ])
        self.dec3 = nn.ModuleList([
            ResidualBlock(self.base_channel * 12, self.base_channel * 4, t_dim, self.cond_out_channel * 8),
            Attention3D(self.base_channel * 4),
            Upsample(self.base_channel * 4)
        ])
        self.dec2 = nn.ModuleList([
            ResidualBlock(self.base_channel * 6, self.base_channel * 2, t_dim, self.cond_out_channel * 4),
            Attention3D(self.base_channel * 2),
            Upsample(self.base_channel * 2)
        ])
        self.dec1 = nn.ModuleList([
            ResidualBlock(self.base_channel * 3, self.base_channel, t_dim, self.cond_out_channel * 2),
            Upsample(self.base_channel)
        ])

        self.out = nn.Conv3d(self.base_channel, self.input, 3, padding=1)

    def forward(self, x, cond, text_features, t):
        cond_list = self.cond_proj(cond)
        t_emb = self.t_embed(t)
        t_emb += self.text_proj(text_features)

        B, T, C, D, H, W = x.shape
        x = x.view(B, -1, D, H, W).contiguous()

        skips = []
        x = self._forward_block(x, t_emb, cond_list[0], self.enc1, skips)
        x = self._forward_block(x, t_emb, cond_list[1], self.enc2, skips)
        x = self._forward_block(x, t_emb, cond_list[2], self.enc3, skips)
        x = self._forward_block(x, t_emb, cond_list[3], self.enc4, skips)

        for layer in self.bneck:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb, cond_list[4])
            else:
                x = layer(x)

        x = self._backward_block(x, skips.pop(), t_emb, cond_list[4], self.dec4)
        x = self._backward_block(x, skips.pop(), t_emb, cond_list[3], self.dec3)
        x = self._backward_block(x, skips.pop(), t_emb, cond_list[2], self.dec2)
        x = self._backward_block(x, skips.pop(), t_emb, cond_list[1], self.dec1)

        motion_out = F.softmax(
            self.out(x).view(B, T, C, D, H, W).view(B, T, C, -1).contiguous(), dim=-1
        ).view(B, T, C, D, H, W).contiguous()

        return motion_out

    def _forward_block(self, x, t_emb, cond, layers, skips):
        for layer in layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb, cond)
            elif isinstance(layer, Attention3D):
                x = layer(x)
            else:
                x = layer(x)
                skips.append(x)
        return x

    def _backward_block(self, x, skip, t_emb, cond, layers):
        for layer in layers:
            if isinstance(layer, Upsample):
                x = layer(x)
            elif isinstance(layer, ResidualBlock):
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb, cond)
            else:
                x = layer(x)
        return x
