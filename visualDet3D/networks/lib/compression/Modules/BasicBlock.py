import torch
from einops import rearrange
from torch import nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, leaky_relu: bool = True):
        super(ResBlock, self).__init__()
        activation_func = nn.LeakyReLU(inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            activation_func,
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            activation_func
        )

    def forward(self, x: torch.Tensor):
        return x + self.block(x)


class LayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=channels)

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b c h w -> b h w c')
        x = rearrange(self.norm(x), 'b h w c -> b c h w')
        return x