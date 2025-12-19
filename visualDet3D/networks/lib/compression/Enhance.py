import torch
import torch.nn as nn

from Modules.BasicBlock import ResBlock

class ResBlocks(nn.Module):
    def __init__(self, channels: int = 32, num_blocks: int = 16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.body = nn.Sequential(*[ResBlock(channels=channels, leaky_relu=True) for _ in range(num_blocks)])
        self.bottom = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=(3, 3), stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor):
        feats_head = self.head(x)
        feats_bottom = feats_head + self.body(feats_head)
        y = x + self.bottom(feats_bottom)
        return y

class EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enhance_l = ResBlocks()
        self.enhance_r = ResBlocks()

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):
        img_enhance_l = self.enhance_l(img_l)
        img_enhance_r = self.enhance_r(img_r)
        img_enhance_l = torch.clip(img_enhance_l, min=0.0, max=1.0)
        img_enhance_r = torch.clip(img_enhance_r, min=0.0, max=1.0)
        return img_enhance_l, img_enhance_r


if __name__ == "__main__":
    from thop import profile
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 3, 512, 512)

    a = EnhanceNet()
    flops, params = profile(a, (x, y), verbose=False)
    print(" Enhancement|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))