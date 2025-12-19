import torch
import torch.nn as nn
from compressai.layers import MaskedConv2d


class ContextInteraction(nn.Module):
    def __init__(self, channels: int = 192):
        super(ContextInteraction, self).__init__()

        self.conv1x1_l = nn.Sequential(
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(1, 1)),

        )
        self.conv1x1_r = nn.Sequential(
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(1, 1)),

        )

        self.fusion_l = nn.Sequential(
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(1, 1))
        )

        self.fusion_r = nn.Sequential(
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 2, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 2, kernel_size=(1, 1)),

        )

        self.inter_l = MaskedConv2d(mask_type="B", in_channels=channels * 2, out_channels=channels * 2,
                                    kernel_size=3, stride=1, padding=1)
        self.inter_r = MaskedConv2d(mask_type="B", in_channels=channels * 2, out_channels=channels * 2,
                                    kernel_size=3, stride=1, padding=1)

    def forward(self, ctx_l: torch.Tensor, ctx_r: torch.Tensor):
        ctx_l = self.conv1x1_l(ctx_l)
        ctx_r = self.conv1x1_r(ctx_r)

        attention_l_r = torch.sigmoid(self.inter_r(self.fusion_r(torch.cat([ctx_r, ctx_l], dim=1))))
        attention_r_l = torch.sigmoid(self.inter_l(self.fusion_l(torch.cat([ctx_l, ctx_r], dim=1))))

        ctx_l_r = attention_l_r * ctx_l
        ctx_r_l = attention_r_l * ctx_r
        return ctx_l_r, ctx_r_l


class GaussianParametersEstimator(nn.Module):
    def __init__(self, channels: int):
        super(GaussianParametersEstimator, self).__init__()
        self.ctx_pred_l = MaskedConv2d(mask_type="A", in_channels=channels, out_channels=2 * channels,
                                       kernel_size=5, stride=1, padding=2)
        self.ctx_pred_r = MaskedConv2d(mask_type="A", in_channels=channels, out_channels=2 * channels,
                                       kernel_size=5, stride=1, padding=2)

        self.ctx_interaction = ContextInteraction(channels=channels)

        self.estimator_l = nn.Sequential(
            nn.Conv2d(6 * channels, 10 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10 * channels // 3, 8 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8 * channels // 3, 2 * channels, kernel_size=(1, 1))
        )
        self.estimator_r = nn.Sequential(
            nn.Conv2d(6 * channels, 10 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10 * channels // 3, 8 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8 * channels // 3, 2 * channels, kernel_size=(1, 1))
        )

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor, hyper_ctx_l: torch.Tensor,
                hyper_ctx_r: torch.Tensor):
        ctx_global_l = self.ctx_pred_l(feats_hat_l)
        ctx_global_r = self.ctx_pred_r(feats_hat_r)

        # context interaction
        ctx_l = torch.cat([hyper_ctx_l, ctx_global_l], dim=1)
        ctx_r = torch.cat([hyper_ctx_r, ctx_global_r], dim=1)
        ctx_l_r, ctx_r_l = self.ctx_interaction(ctx_l, ctx_r)

        param_l = self.estimator_l(torch.cat([ctx_l, ctx_r_l], dim=1))
        param_r = self.estimator_r(torch.cat([ctx_r, ctx_l_r], dim=1))
        return param_l, param_r


class GaussianParametersEstimator_Ablation(nn.Module):
    def __init__(self, channels: int):
        super(GaussianParametersEstimator_Ablation, self).__init__()
        self.ctx_pred_l = MaskedConv2d(mask_type="A", in_channels=channels, out_channels=2 * channels,
                                       kernel_size=5, stride=1, padding=2)
        self.ctx_pred_r = MaskedConv2d(mask_type="A", in_channels=channels, out_channels=2 * channels,
                                       kernel_size=5, stride=1, padding=2)

        #self.ctx_interaction = ContextInteraction(channels=channels)

        self.estimator_l = nn.Sequential(
            nn.Conv2d(4 * channels, 10 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10 * channels // 3, 8 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8 * channels // 3, 2 * channels, kernel_size=(1, 1))
        )
        self.estimator_r = nn.Sequential(
            nn.Conv2d(4 * channels, 10 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10 * channels // 3, 8 * channels // 3, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8 * channels // 3, 2 * channels, kernel_size=(1, 1))
        )

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor, hyper_ctx_l: torch.Tensor,
                hyper_ctx_r: torch.Tensor):
        ctx_global_l = self.ctx_pred_l(feats_hat_l)
        ctx_global_r = self.ctx_pred_r(feats_hat_r)

        # context interaction
        ctx_l = torch.cat([hyper_ctx_l, ctx_global_l], dim=1)
        ctx_r = torch.cat([hyper_ctx_r, ctx_global_r], dim=1)
        #ctx_l_r, ctx_r_l = self.ctx_interaction(ctx_l, ctx_r)

        param_l = self.estimator_l(ctx_l)
        param_r = self.estimator_r(ctx_r)
        return param_l, param_r
