import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_


class SliceCrossAttention(nn.Module):
    def __init__(self, channels: int, slice_height: int, num_heads: int, base_view: str, slice_width: int, shift: int, top_k: int):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.shift = shift
        self.base_view = base_view
        self.top_k = top_k

        self.slice_height = slice_height
        self.slice_width = slice_width

        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = self.init_relative_pos()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, C, H, W = query.shape

        query, key, value, mask = self.windows_partition(query=query, key=key, value=value)
        mask = mask.view(-1, 1, 1, self.slice_height * self.slice_width)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.slice_height * self.slice_width,
                                                             self.slice_height * self.slice_width, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(dim=0).contiguous()

        query = query * self.scale
        attn = (query @ key.transpose(-2, -1))

        index = torch.topk(attn, k=self.top_k, dim=-1, largest=True)[1]
        knn_mask = torch.scatter(torch.zeros_like(attn), dim=-1, index=index, value=1.0)

        attn = self.softmax(attn + relative_position_bias) * mask * knn_mask

        out = rearrange(attn @ value, '(b n m) k (h w) c -> b (k c) (n h) (m w)',
                        n=H // self.slice_height, m=W // self.slice_width, h=self.slice_height)
        return out

    def windows_partition(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        B, C, H, W = query.shape
        assert H % self.slice_height == 0 and W % self.slice_width == 0, "invalid query size {0}x{1}".format(str(H), str(W))
        B, C, H, W = key.shape
        assert H % self.slice_height == 0 and W % self.slice_width == 0, "invalid key size {0}x{1}".format(str(H), str(W))
        B, C, H, W = value.shape
        assert H % self.slice_height == 0 and W % self.slice_width == 0, "invalid value size {0}x{1}".format(str(H), str(W))

        query = rearrange(query, 'b (k c) (n h) (m w) -> (b n m) k (h w) c',
                          k=self.num_heads, h=self.slice_height, w=self.slice_width)

        mask = torch.ones(B, H, W).to(key.device)
        # shift
        if self.shift > 0:
            if self.base_view == "left":
                mask[:, :, :self.shift] = 0.0
                key = torch.roll(key, shifts=self.shift, dims=-1)
                value = torch.roll(value, shifts=self.shift, dims=-1)
            else:
                mask[:, :, -self.shift:] = 0.0
                key = torch.roll(key, shifts=-self.shift, dims=-1)
                value = torch.roll(value, shifts=-self.shift, dims=-1)
        mask = rearrange(mask, 'b (n s) (m q) -> (b n m) (s q)', s=self.slice_height, q=self.slice_width)

        key = rearrange(key, 'b (k c) (n h) (m w) -> (b n m) k (h w) c', k=self.num_heads, h=self.slice_height, w=self.slice_width)
        value = rearrange(value, 'b (k c) (n h) (m w) -> (b n m) k (h w) c', k=self.num_heads, h=self.slice_height, w=self.slice_width)

        return query, key, value, mask

    def init_relative_pos(self):
        coords = torch.stack(torch.meshgrid([torch.arange(self.slice_height), torch.arange(self.slice_width)]))
        coords_flatten = torch.flatten(coords, start_dim=1)
        ref_coords = torch.stack(torch.meshgrid([torch.arange(self.slice_height), torch.arange(self.slice_width)]))
        ref_coords_flatten = torch.flatten(ref_coords, start_dim=1)
        relative_coords = coords_flatten[:, :, None] - ref_coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.slice_height - 1
        relative_coords[:, :, 1] += self.slice_width - 1
        relative_coords[:, :, 0] *= 2 * self.slice_width - 1
        relative_position_index = relative_coords.sum(dim=-1)
        self.register_buffer("relative_position_index", relative_position_index)
        relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.slice_height - 1) * (2 * self.slice_width - 1), self.num_heads)
        )
        trunc_normal_(relative_position_bias_table, std=.02)
        return relative_position_bias_table

    def flops(self):
        N = self.slice_width * self.slice_height
        flops = 0
        # A = q @ k
        flops += N * N * self.channels
        # o = A @ v
        flops += N * N * self.channels
        return flops


if __name__ == "__main__":
    H = [256, 128, 64, 32]
    W = [256, 128, 64, 32]
    slice_height = 4
    slice_width = 16

    a = SliceCrossAttention(channels=128, slice_height=slice_height, num_heads=8,
                            base_view="left", slice_width=slice_width, shift=0,
                            top_k=slice_height * slice_width // 2)
    flops = 0
    for i in range(4):
        flops_one_tr = H[i] * W[i] // slice_height // slice_width * a.flops()
        flops += 4 * flops_one_tr
    print(flops * 2 / 1e9)
