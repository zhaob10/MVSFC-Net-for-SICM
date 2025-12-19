import sys
sys.path.append('/media/D/visualDet3D-master/')
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.models import CompressionModel
from compressai.models.utils import update_registered_buffers

from visualDet3D.networks.lib.compression.Modules.AutoEncoder import Encoder, Decoder, HyperEncoder, HyperDecoder, \
                                                                     Encoder_Ablation, Decoder_Ablation, Encoder_Ablation_2, Decoder_Ablation_2, \
                                                                     FeatureCombine, FeatureSynthesis, HyperDecoder_Kim, HyperEncoder_Kim, \
                                                                     Encoder_Ablation_sfatten, Decoder_Ablation_sfatten
from visualDet3D.networks.lib.compression.Modules.EntropyModel import GaussianParametersEstimator, GaussianParametersEstimator_Ablation
from visualDet3D.networks.lib.compression.Modules.Utils import get_scale_table


class Codec(CompressionModel):
    def __init__(self):
        super(Codec, self).__init__(entropy_bottleneck_channels=128)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()

        self.param_estimator = GaussianParametersEstimator(channels=192)

        # self.encoder = FeatureCombine()
        # self.decoder = FeatureSynthesis()

        # self.hyper_encoder = HyperEncoder_Kim()
        # self.hyper_decoder = HyperDecoder_Kim()

        # self.param_estimator = GaussianParametersEstimator(channels=192)

        self.entropy_bottleneck_r = EntropyBottleneck(channels=128)

        self.gaussian_l = GaussianConditional(None)
        self.gaussian_r = GaussianConditional(None)

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):
        feats_l, feats_r = self.encoder(img_l, img_r)
        hyper_l, hyper_r = self.hyper_encoder(feats_l, feats_r) # [4, 256, 12, 40]

        hyper_hat_l, hyper_likelihoods_l = self.entropy_bottleneck(hyper_l)
        hyper_hat_r, hyper_likelihoods_r = self.entropy_bottleneck_r(hyper_r)

        hyper_ctx_l, hyper_ctx_r = self.hyper_decoder(hyper_hat_l, hyper_hat_r)

        feats_hat_l = self.gaussian_l.quantize(feats_l, "noise" if self.training else "dequantize")
        feats_hat_r = self.gaussian_r.quantize(feats_r, "noise" if self.training else "dequantize")

        params_l, params_r = self.param_estimator(feats_hat_l, feats_hat_r, hyper_ctx_l, hyper_ctx_r)
        scales_l, means_l = params_l.chunk(2, 1)
        scales_r, means_r = params_r.chunk(2, 1)

        _, feat_likelihoods_l = self.gaussian_l(feats_l, F.relu(scales_l), means=means_l)
        _, feat_likelihoods_r = self.gaussian_r(feats_r, F.relu(scales_r), means=means_r)

        img_hat_l, img_hat_r = self.decoder(feats_hat_l, feats_hat_r)

        # img_hat_l = torch.clamp(img_hat_l, min=0.0, max=1.0) # 这里可能需要去除
        # img_hat_r = torch.clamp(img_hat_r, min=0.0, max=1.0) # 这里可能需要去除

        return {
            'img_hat': [img_hat_l, img_hat_r],
            'feat_likelihoods': [feat_likelihoods_l, feat_likelihoods_r],
            'hyper_likelihoods': [hyper_likelihoods_l, hyper_likelihoods_r]
        }

    @torch.no_grad()
    def compress(self, img_left: torch.Tensor, img_right: torch.Tensor):

        feats_l, feats_r = self.encoder(img_left, img_right)

        hyper_l, hyper_r = self.hyper_encoder(feats_l, feats_r)

        strings_hyper_l = self.entropy_bottleneck.compress(hyper_l)
        strings_hyper_r = self.entropy_bottleneck_r.compress(hyper_r)

        hyper_hat_l = self.entropy_bottleneck.decompress(strings_hyper_l, size=hyper_l.size()[-2:])
        hyper_hat_r = self.entropy_bottleneck_r.decompress(strings_hyper_r, size=hyper_r.size()[-2:])

        hyper_ctx_l, hyper_ctx_r = self.hyper_decoder(hyper_hat_l, hyper_hat_r)

        strings_l, strings_r = self.compress_autoregressive(feats_l=feats_l, feats_r=feats_r,
                                                            hyper_ctx_l=hyper_ctx_l, hyper_ctx_r=hyper_ctx_r)

        return {"strings": [strings_l, strings_r, strings_hyper_l, strings_hyper_r], "shape": hyper_l.size()[-2:]}

    @torch.no_grad()
    def compress_autoregressive(self, feats_l: torch.Tensor, feats_r: torch.Tensor,
                                hyper_ctx_l: torch.Tensor, hyper_ctx_r: torch.Tensor):
        device = next(self.parameters()).device

        encoder_l = BufferedRansEncoder()
        encoder_r = BufferedRansEncoder()

        cdf_l = self.gaussian_l.quantized_cdf.tolist()
        cdf_r = self.gaussian_r.quantized_cdf.tolist()
        cdf_lengths_l = self.gaussian_l.cdf_length.tolist()
        cdf_lengths_r = self.gaussian_r.cdf_length.tolist()
        offsets_l = self.gaussian_l.offset.tolist()
        offsets_r = self.gaussian_r.offset.tolist()

        B, C, H, W = feats_l.shape

        auto_kernel_size = self.param_estimator.ctx_pred_l.weight.shape[-1]
        auto_padding = (auto_kernel_size - 1) // 2
        inter_kernel_size = self.param_estimator.ctx_interaction.inter_l.weight.shape[-1]
        inter_padding = (inter_kernel_size - 1) // 2

        auto_weight_l = self.param_estimator.ctx_pred_l.weight * self.param_estimator.ctx_pred_l.mask
        auto_weight_r = self.param_estimator.ctx_pred_r.weight * self.param_estimator.ctx_pred_r.mask
        auto_bias_l = self.param_estimator.ctx_pred_l.bias
        auto_bias_r = self.param_estimator.ctx_pred_r.bias

        inter_weight_l = self.param_estimator.ctx_interaction.inter_l.weight * self.param_estimator.ctx_interaction.inter_l.mask
        inter_weight_r = self.param_estimator.ctx_interaction.inter_r.weight * self.param_estimator.ctx_interaction.inter_r.mask
        inter_bias_l = self.param_estimator.ctx_interaction.inter_l.bias
        inter_bias_r = self.param_estimator.ctx_interaction.inter_r.bias

        feats_hat_l = F.pad(feats_l, pad=(auto_padding, auto_padding, auto_padding, auto_padding)).to(device)
        feats_hat_r = F.pad(feats_r, pad=(auto_padding, auto_padding, auto_padding, auto_padding)).to(device)

        feats_quantized_l = torch.zeros_like(feats_l, device=device)
        feats_quantized_r = torch.zeros_like(feats_r, device=device)

        scales_l = torch.zeros_like(feats_l, device=device)
        scales_r = torch.zeros_like(feats_r, device=device)

        # [0: 2C] is the hyper ctx, [2C: 4C] is the autoregressive ctx
        ctx_l_pad = F.pad(torch.cat([hyper_ctx_l, torch.zeros(B, 4 * C, H, W, device=device)], dim=1),
                          pad=(inter_padding, inter_padding, inter_padding, inter_padding))
        ctx_r_pad = F.pad(torch.cat([hyper_ctx_r, torch.zeros(B, 4 * C, H, W, device=device)], dim=1),
                          pad=(inter_padding, inter_padding, inter_padding, inter_padding)).to(device)
        for h in range(H):
            for w in range(W):
                # generate autoregressive context
                patch_hat_l = feats_hat_l[:, :, h: h + auto_kernel_size, w: w + auto_kernel_size]
                patch_hat_r = feats_hat_r[:, :, h: h + auto_kernel_size, w: w + auto_kernel_size]

                auto_ctx_l = F.conv2d(patch_hat_l, weight=auto_weight_l, bias=auto_bias_l)
                auto_ctx_r = F.conv2d(patch_hat_r, weight=auto_weight_r, bias=auto_bias_r)

                ctx_l_pad[:, 2 * C: 4 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = auto_ctx_l
                ctx_r_pad[:, 2 * C: 4 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = auto_ctx_r

                # generate inter-view context
                patch_ctx_l = ctx_l_pad[:, : 4 * C, h: h + inter_kernel_size, w: w + inter_kernel_size]
                patch_ctx_r = ctx_r_pad[:, : 4 * C, h: h + inter_kernel_size, w: w + inter_kernel_size]

                patch_ctx_l_reduce = self.param_estimator.ctx_interaction.conv1x1_l(patch_ctx_l)
                patch_ctx_r_reduce = self.param_estimator.ctx_interaction.conv1x1_r(patch_ctx_r)

                attention_l_r = torch.sigmoid(F.conv2d(self.param_estimator.ctx_interaction.fusion_r(
                    torch.cat([patch_ctx_r_reduce, patch_ctx_l_reduce], dim=1)),
                                                       weight=inter_weight_r, bias=inter_bias_r))
                attention_r_l = torch.sigmoid(F.conv2d(self.param_estimator.ctx_interaction.fusion_l(
                    torch.cat([patch_ctx_l_reduce, patch_ctx_r_reduce], dim=1)),
                                                       weight=inter_weight_l, bias=inter_bias_l))

                inter_ctx_l = attention_r_l * patch_ctx_r_reduce[:, :, inter_padding: inter_padding + 1,
                                              inter_padding: inter_padding + 1]
                inter_ctx_r = attention_l_r * patch_ctx_l_reduce[:, :, inter_padding: inter_padding + 1,
                                              inter_padding: inter_padding + 1]

                ctx_l_pad[:, 4 * C: 6 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = inter_ctx_l
                ctx_r_pad[:, 4 * C: 6 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = inter_ctx_r

                # estimate entropy parameters
                ctx_l = ctx_l_pad[:, :, h + inter_padding: h + inter_padding + 1,
                                  w + inter_padding: w + inter_padding + 1]
                ctx_r = ctx_r_pad[:, :, h + inter_padding: h + inter_padding + 1,
                                  w + inter_padding: w + inter_padding + 1]

                params_l = self.param_estimator.estimator_l(ctx_l).squeeze(dim=3).squeeze(dim=2)
                params_r = self.param_estimator.estimator_r(ctx_r).squeeze(dim=3).squeeze(dim=2)
                scale_l, means_l = params_l.chunk(chunks=2, dim=1)
                scale_r, means_r = params_r.chunk(chunks=2, dim=1)

                scales_l[:, :, h, w] = scale_l
                scales_r[:, :, h, w] = scale_r

                # quantization
                symbol_l = patch_hat_l[:, :, auto_padding, auto_padding]
                symbol_r = patch_hat_r[:, :, auto_padding, auto_padding]

                symbol_hat_l = self.gaussian_l.quantize(inputs=symbol_l, mode="symbols", means=means_l)
                symbol_hat_r = self.gaussian_r.quantize(inputs=symbol_r, mode="symbols", means=means_r)

                feats_quantized_l[:, :, h, w] = symbol_hat_l
                feats_quantized_r[:, :, h, w] = symbol_hat_r

                feats_hat_l[:, :, h + auto_padding, w + auto_padding] = symbol_hat_l + means_l
                feats_hat_r[:, :, h + auto_padding, w + auto_padding] = symbol_hat_r + means_r

        # entropy coding
        indexes_l = self.gaussian_l.build_indexes(F.relu(scales_l.permute(0, 2, 3, 1))).cpu().reshape(-1).int().tolist()
        indexes_r = self.gaussian_r.build_indexes(F.relu(scales_r.permute(0, 2, 3, 1))).cpu().reshape(-1).int().tolist()

        symbols_l = feats_quantized_l.permute(0, 2, 3, 1).cpu().reshape(-1).int().tolist()
        symbols_r = feats_quantized_r.permute(0, 2, 3, 1).cpu().reshape(-1).int().tolist()

        encoder_l.encode_with_indexes(symbols_l, indexes_l, cdf_l, cdf_lengths_l, offsets_l)
        encoder_r.encode_with_indexes(symbols_r, indexes_r, cdf_r, cdf_lengths_r, offsets_r)
        strings_l = encoder_l.flush()
        strings_r = encoder_r.flush()

        return [strings_l, ], [strings_r, ]

    @torch.no_grad()
    def decompress(self, strings: list, shape: list):
        device = next(self.parameters()).device
        assert isinstance(strings, list) and len(strings) == 4
        strings_l, strings_r, strings_hyper_l, strings_hyper_r = strings

        hyper_hat_l = self.entropy_bottleneck.decompress(strings_hyper_l, size=shape)
        hyper_hat_r = self.entropy_bottleneck_r.decompress(strings_hyper_r, size=shape)

        hyper_ctx_l, hyper_ctx_r = self.hyper_decoder(hyper_hat_l.to(device), hyper_hat_r.to(device))

        feats_hat_l, feats_hat_r = self.decompress_autoregressive(strings_l=strings_l[0], strings_r=strings_r[0],
                                                                  hyper_ctx_l=hyper_ctx_l, hyper_ctx_r=hyper_ctx_r)
        img_hat_l, img_hat_r = self.decoder(feats_hat_l.to(device), feats_hat_r.to(device))

        # img_hat_l = torch.clamp(img_hat_l, min=0.0, max=1.0)
        # img_hat_r = torch.clamp(img_hat_r, min=0.0, max=1.0)

        return {
            "img_hat": [img_hat_l, img_hat_r]
        }

    @torch.no_grad()
    def decompress_autoregressive(self, strings_l: bytes, strings_r: bytes,
                                  hyper_ctx_l: torch.Tensor, hyper_ctx_r: torch.Tensor):

        device = next(self.parameters()).device

        decoder_l = RansDecoder()
        decoder_r = RansDecoder()
        decoder_l.set_stream(strings_l)
        decoder_r.set_stream(strings_r)

        cdf_l = self.gaussian_l.quantized_cdf.tolist()
        cdf_r = self.gaussian_r.quantized_cdf.tolist()
        cdf_lengths_l = self.gaussian_l.cdf_length.tolist()
        cdf_lengths_r = self.gaussian_r.cdf_length.tolist()
        offsets_l = self.gaussian_l.offset.tolist()
        offsets_r = self.gaussian_r.offset.tolist()

        B, C, H, W = hyper_ctx_l.shape
        C = C // 2  # note that

        auto_kernel_size = self.param_estimator.ctx_pred_l.weight.shape[-1]
        auto_padding = (auto_kernel_size - 1) // 2
        inter_kernel_size = self.param_estimator.ctx_interaction.inter_l.weight.shape[-1]
        inter_padding = (inter_kernel_size - 1) // 2

        auto_weight_l = self.param_estimator.ctx_pred_l.weight * self.param_estimator.ctx_pred_l.mask
        auto_weight_r = self.param_estimator.ctx_pred_r.weight * self.param_estimator.ctx_pred_r.mask
        auto_bias_l = self.param_estimator.ctx_pred_l.bias
        auto_bias_r = self.param_estimator.ctx_pred_r.bias

        inter_weight_l = self.param_estimator.ctx_interaction.inter_l.weight * self.param_estimator.ctx_interaction.inter_l.mask
        inter_weight_r = self.param_estimator.ctx_interaction.inter_r.weight * self.param_estimator.ctx_interaction.inter_r.mask
        inter_bias_l = self.param_estimator.ctx_interaction.inter_l.bias
        inter_bias_r = self.param_estimator.ctx_interaction.inter_r.bias

        feats_hat_l = F.pad(torch.zeros(B, C, H, W, device=device), pad=(auto_padding, auto_padding, auto_padding, auto_padding))
        feats_hat_r = F.pad(torch.zeros(B, C, H, W, device=device), pad=(auto_padding, auto_padding, auto_padding, auto_padding))

        # [0: 2C] is the hyper ctx, [2C: 4C] is the autoregressive ctx, [4C: 6C] is the inter-view ctx
        ctx_l_pad = F.pad(torch.cat([hyper_ctx_l, torch.zeros(B, 4 * C, H, W, device=device)], dim=1),
                          pad=(inter_padding, inter_padding, inter_padding, inter_padding))
        ctx_r_pad = F.pad(torch.cat([hyper_ctx_r, torch.zeros(B, 4 * C, H, W, device=device)], dim=1),
                          pad=(inter_padding, inter_padding, inter_padding, inter_padding))

        for h in range(H):
            for w in range(W):
                # generate autoregressive context
                patch_hat_l = feats_hat_l[:, :, h: h + auto_kernel_size, w: w + auto_kernel_size]
                patch_hat_r = feats_hat_r[:, :, h: h + auto_kernel_size, w: w + auto_kernel_size]

                auto_ctx_l = F.conv2d(patch_hat_l, weight=auto_weight_l, bias=auto_bias_l)
                auto_ctx_r = F.conv2d(patch_hat_r, weight=auto_weight_r, bias=auto_bias_r)

                ctx_l_pad[:, 2 * C: 4 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = auto_ctx_l
                ctx_r_pad[:, 2 * C: 4 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = auto_ctx_r

                # generate inter-view context
                patch_ctx_l = ctx_l_pad[:, : 4 * C, h: h + inter_kernel_size, w: w + inter_kernel_size]
                patch_ctx_r = ctx_r_pad[:, : 4 * C, h: h + inter_kernel_size, w: w + inter_kernel_size]

                patch_ctx_l = self.param_estimator.ctx_interaction.conv1x1_l(patch_ctx_l)
                patch_ctx_r = self.param_estimator.ctx_interaction.conv1x1_r(patch_ctx_r)

                attention_l_r = torch.sigmoid(F.conv2d(
                    self.param_estimator.ctx_interaction.fusion_r(torch.cat([patch_ctx_r, patch_ctx_l], dim=1)),
                    weight=inter_weight_r, bias=inter_bias_r))
                attention_r_l = torch.sigmoid(F.conv2d(
                    self.param_estimator.ctx_interaction.fusion_l(torch.cat([patch_ctx_l, patch_ctx_r], dim=1)),
                    weight=inter_weight_l, bias=inter_bias_l))

                inter_ctx_l = attention_r_l * patch_ctx_r[:, :, inter_padding: inter_padding + 1,
                                                          inter_padding: inter_padding + 1]
                inter_ctx_r = attention_l_r * patch_ctx_l[:, :, inter_padding: inter_padding + 1,
                                                          inter_padding: inter_padding + 1]

                ctx_l_pad[:, 4 * C: 6 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = inter_ctx_l
                ctx_r_pad[:, 4 * C: 6 * C, h + inter_padding: h + inter_padding + 1,
                          w + inter_padding: w + inter_padding + 1] = inter_ctx_r

                # estimate entropy parameters
                ctx_l = ctx_l_pad[:, :, h + inter_padding: h + inter_padding + 1, w + inter_padding: w + inter_padding + 1]
                ctx_r = ctx_r_pad[:, :, h + inter_padding: h + inter_padding + 1, w + inter_padding: w + inter_padding + 1]

                params_l = self.param_estimator.estimator_l(ctx_l)
                params_r = self.param_estimator.estimator_r(ctx_r)
                scales_l, means_l = params_l.chunk(2, 1)
                scales_r, means_r = params_r.chunk(2, 1)

                indexes_l = self.gaussian_l.build_indexes(F.relu(scales_l))
                indexes_r = self.gaussian_r.build_indexes(F.relu(scales_r))

                rv_l = decoder_l.decode_stream(indexes_l.squeeze().cpu().tolist(), cdf_l, cdf_lengths_l, offsets_l)
                rv_l = torch.Tensor(rv_l).reshape(1, -1, 1, 1).to(device)
                rv_l = self.gaussian_l.dequantize(rv_l, means_l)

                rv_r = decoder_r.decode_stream(indexes_r.squeeze().cpu().tolist(), cdf_r, cdf_lengths_r, offsets_r)
                rv_r = torch.Tensor(rv_r).reshape(1, -1, 1, 1).to(device)
                rv_r = self.gaussian_r.dequantize(rv_r, means_r)

                feats_hat_l[:, :, h + auto_padding: h + auto_padding + 1, w + auto_padding: w + auto_padding + 1] = rv_l
                feats_hat_r[:, :, h + auto_padding: h + auto_padding + 1, w + auto_padding: w + auto_padding + 1] = rv_r

        feats_hat_l = F.pad(feats_hat_l, pad=(-auto_padding, -auto_padding, -auto_padding, -auto_padding))
        feats_hat_r = F.pad(feats_hat_r, pad=(-auto_padding, -auto_padding, -auto_padding, -auto_padding))

        return feats_hat_l, feats_hat_r

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_l,
            "gaussian_l",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_r,
            "gaussian_r",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck_r,
            "entropy_bottleneck_r",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table: torch.Tensor = None, force: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_l.update_scale_table(scale_table, force=force)
        updated |= self.gaussian_r.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


class Codec_Ablation(CompressionModel):
    def __init__(self):
        super(Codec_Ablation, self).__init__(entropy_bottleneck_channels=128)

        self.encoder = Encoder_Ablation()
        self.decoder = Decoder_Ablation()

        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()

        # self.param_estimator = GaussianParametersEstimator_Ablation(channels=192)
        self.param_estimator = GaussianParametersEstimator(channels=192)

        self.entropy_bottleneck_r = EntropyBottleneck(channels=128)

        self.gaussian_l = GaussianConditional(None)
        self.gaussian_r = GaussianConditional(None)

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):
        feats_l, feats_r = self.encoder(img_l, img_r)
        hyper_l, hyper_r = self.hyper_encoder(feats_l, feats_r) # [4, 256, 12, 40]

        hyper_hat_l, hyper_likelihoods_l = self.entropy_bottleneck(hyper_l)
        hyper_hat_r, hyper_likelihoods_r = self.entropy_bottleneck_r(hyper_r)

        hyper_ctx_l, hyper_ctx_r = self.hyper_decoder(hyper_hat_l, hyper_hat_r)

        feats_hat_l = self.gaussian_l.quantize(feats_l, "noise" if self.training else "dequantize")
        feats_hat_r = self.gaussian_r.quantize(feats_r, "noise" if self.training else "dequantize")

        params_l, params_r = self.param_estimator(feats_hat_l, feats_hat_r, hyper_ctx_l, hyper_ctx_r)
        scales_l, means_l = params_l.chunk(2, 1)
        scales_r, means_r = params_r.chunk(2, 1)

        _, feat_likelihoods_l = self.gaussian_l(feats_l, F.relu(scales_l), means=means_l)
        _, feat_likelihoods_r = self.gaussian_r(feats_r, F.relu(scales_r), means=means_r)

        img_hat_l, img_hat_r = self.decoder(feats_hat_l, feats_hat_r)

        # img_hat_l = torch.clamp(img_hat_l, min=0.0, max=1.0) # 这里可能需要去除
        # img_hat_r = torch.clamp(img_hat_r, min=0.0, max=1.0) # 这里可能需要去除

        return {
            'img_hat': [img_hat_l, img_hat_r],
            'feat_likelihoods': [feat_likelihoods_l, feat_likelihoods_r],
            'hyper_likelihoods': [hyper_likelihoods_l, hyper_likelihoods_r]
        }
        

class Codec_Ablation_2(Codec):
    def __init__(self):
        super(Codec_Ablation_2, self).__init__()
        
        self.encoder = Encoder_Ablation_2()
        self.decoder = Decoder_Ablation_2()
    
class Codec_Ablation_sfatten(Codec):
    def __init__(self):
        super(Codec_Ablation_sfatten, self).__init__()
        
        self.encoder = Encoder_Ablation_sfatten()
        self.decoder = Decoder_Ablation_sfatten()


from visualDet3D.networks.lib.compression.feature_compression_comparison import CodecSep

class Codec_MFCNet(nn.Module):
    def __init__(self, M = 192):
        super(Codec_MFCNet, self).__init__()
        
        self.codec_l = CodecSep(N = [64, 128, 256], M = M) # [[4, 64, 72, 320], [4, 128, 36, 160], [4, 256, 18, 80]], 12, 40
        self.codec_r = CodecSep(N = [64, 128, 256], M = M) # [[4, 64, 72, 320], [4, 128, 36, 160], [4, 256, 18, 80]], 12, 40

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):

        result_l = self.codec_l(img_l)
        result_r = self.codec_r(img_r)
    
        return {
            'img_hat': [result_l['img_hat'], result_r['img_hat']],
            'feat_likelihoods': [result_l['feat_likelihoods'], result_r['feat_likelihoods']],
            'hyper_likelihoods': [result_l['hyper_likelihoods'], result_r['hyper_likelihoods']]
        }

    def aux_loss(self):
        return self.codec_l.aux_loss() + self.codec_r.aux_loss()

    def update(self):
        self.codec_l.update()
        self.codec_r.update()
    
    def compress(self, img_l: torch.Tensor, img_r: torch.Tensor):
        
        result_l_1, result_l_2 = self.codec_l.compress(img_l)
        result_r_1, result_r_2 = self.codec_r.compress(img_r)
        
        strings_list = [result_l_1["strings"], result_l_2["strings"], result_r_1["strings"], result_r_2["strings"]]
        shape_lsit   = [result_l_1["shape"],   result_l_2["shape"],   result_r_1["shape"],   result_r_2["shape"]]
        return {
            "strings": strings_list, 
            "shape": shape_lsit
        }

    def decompress(self, strings: list, shape: list):
        
        string_l_1, string_l_2, string_r_1, string_r_2 = strings
        shape_l_1,  shape_l_2,  shape_r_1,  shape_r_2  = shape
                    
        img_hat_l = self.codec_l.decompress(string_l_1, shape_l_1, string_l_2, shape_l_2)['img_hat']
        img_hat_r = self.codec_r.decompress(string_r_1, shape_r_1, string_r_2, shape_r_2)['img_hat']
        
        return {
            "img_hat": [img_hat_l, img_hat_r]
        }

if __name__ == "__main__":
    from thop import profile
    import sys
    sys.path.append('/media/D/visualDet3D-master/')

    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 3, 512, 512)
    a = Codec()

    flops, params = profile(a, (x, y), verbose=False)
    print(" Codec|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))


