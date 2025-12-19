from turtle import forward
import torch
from torch import nn as nn
import torch.nn.functional as F

try:
    from Modules.Attention import SliceCrossAttention
    from Modules.BasicBlock import ResBlock, LayerNorm
except:
    from .Attention import SliceCrossAttention
    from .BasicBlock import ResBlock, LayerNorm

class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_, input2):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input2).reshape(n, self.key_channels, h * w)
        values = self.values(input2).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class ViewAwareTransform(nn.Module):
    def __init__(self, input_channels:int, head_count:int):
        super(ViewAwareTransform, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels=2*input_channels, out_channels=input_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Sigmoid()
        ) 

        self.bottle_block = nn.Sequential(
            nn.Conv2d(in_channels=2*input_channels, out_channels=input_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        ) 

        self.nonlocal_block = EfficientAttention(in_channels=input_channels, key_channels=input_channels, head_count=head_count, value_channels=input_channels)

    def forward(self, feature_base, feature_view):

        view_base = self.nonlocal_block(feature_base, feature_view)
        view_base_enhanced = torch.mul(view_base, self.blocks(torch.cat([view_base, feature_base], dim=1)))
        output = self.bottle_block(torch.cat([feature_base, view_base_enhanced], dim=1))

        return output

class InteractionModule(nn.Module):
    def __init__(self, channels: int, head_count:int=8):
        super(InteractionModule, self).__init__()

        self.res_blocks_l = nn.Sequential(
            ResBlock(channels=channels),
            ResBlock(channels=channels)
        )
        self.res_blocks_r = nn.Sequential(
            ResBlock(channels=channels),
            ResBlock(channels=channels)
        )

        self.transform_l = ViewAwareTransform(input_channels=channels, head_count=head_count)
        self.transform_r = ViewAwareTransform(input_channels=channels, head_count=head_count)

    def forward(self, feats_l: torch.Tensor, feats_r: torch.Tensor):
        feats_l_ = self.res_blocks_l(feats_l)
        feats_r_ = self.res_blocks_r(feats_r)

        feats_l = feats_l - self.transform_l(feature_base=feats_l_, feature_view=feats_r_)
        feats_r = feats_r - self.transform_r(feature_base=feats_r_, feature_view=feats_l_)

        return feats_l, feats_r
