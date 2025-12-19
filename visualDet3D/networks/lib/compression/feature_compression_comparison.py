

import torch
import torch.nn as nn
from compressai.layers import GDN, MaskedConv2d
from compressai.models import CompressionModel
from compressai.models.priors import ScaleHyperprior, JointAutoregressiveHierarchicalPriors
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
import torch.nn.functional as F
from compressai.models.utils import update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from visualDet3D.networks.lib.compression.feature_compression_modules import *

import math
import warnings


from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN,
)

# •	512 × 34×19
# •	256 × 68×38
# •	128 × 136×76
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min_: float = SCALES_MIN, max_: float = SCALES_MAX, levels: int = SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min_), math.log(max_), levels))


class CodecBase1(CompressionModel):
    def __init__(self, N, M):
        super(CodecBase1, self).__init__(entropy_bottleneck_channels=M)

        self.encoder = EncoderBase1(N = N, M = M)
        #self.decoder = DecoderBase1()
        self.decoder = DecoderBase2(N = N*2, M = M)


        self.hyper_encoder = HyperEncoder(M = M)
        self.hyper_decoder = HyperDecoder(M = M)

        self.entropy_bottleneck = EntropyBottleneck(channels=M)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        self.gaussian = GaussianConditional(None)

        self.M = M

    def forward(self, img: torch.Tensor):
        feats = self.encoder(img)
        hyper = self.hyper_encoder(feats)

        hyper_hat, hyper_likelihoods = self.entropy_bottleneck(hyper)
        hyper_ctx = self.hyper_decoder(hyper_hat)

        feats_hat = self.gaussian.quantize(feats, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(feats_hat)
        params = self.entropy_parameters(torch.cat((hyper_ctx, ctx_params), dim=1))

        scales, means = params.chunk(2, 1)
        _, feat_likelihoods = self.gaussian(feats, F.relu(scales), means=means)

        img_hat = self.decoder(feats_hat)
        return {
            'img_hat': img_hat,
            'feat_likelihoods': feat_likelihoods,
            'hyper_likelihoods': hyper_likelihoods
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian,
            "gaussian",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)


    def update(self, scale_table: torch.Tensor = None, force: bool = False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.encoder(x)
        z = self.hyper_encoder(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.hyper_decoder(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian.quantized_cdf.tolist()
        cdf_lengths = self.gaussian.cdf_length.tolist()
        offsets = self.gaussian.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.hyper_decoder(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

        x_hat = self.decoder(y_hat)
        return {
            'img_hat': x_hat,
            'feat_likelihoods': {torch.tensor(0)},
            'hyper_likelihoods': {torch.tensor(0)}
        }

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian.quantized_cdf.tolist()
        cdf_lengths = self.gaussian.cdf_length.tolist()
        offsets = self.gaussian.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

class CodecBase3(CodecBase1):
    def __init__(self, N, M):
        super(CodecBase3, self).__init__(N = N, M = M)
        self.encoder = EncoderBase3(N = N, M = M)
        self.decoder = DecoderBase3(N = N, M = M)

class CodecSep(nn.Module):
    def __init__(self, N: list, M=128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.codec_1 = CodecBase1(N = N[0], M = M)
        #self.codec_2 = CodecBase2()
        self.codec_3 = CodecBase3(N = N[2], M = M)

        self.prmodel1 = prmodel()
    
    def forward(self, img: torch.Tensor):
        result1 = self.codec_1(img[0])
        recon1 = self.prmodel1(result1['img_hat'])
        #result2 = self.codec_2(img[1])
        result3 = self.codec_3(img[2])

        img_hat = [recon1, result1['img_hat'], result3['img_hat']]
        feat_likelihoods = [result1['feat_likelihoods'], 
                            result3['feat_likelihoods'], 
                            ]
        hyper_likelihoods = [result1['hyper_likelihoods'], 
                            result3['hyper_likelihoods'], 
                            ]

        # img_hat = [result1['img_hat'], result2['img_hat'], result3['img_hat']]
        # feat_likelihoods = [result1['feat_likelihoods'], 
        #                     result2['feat_likelihoods'], 
        #                     result3['feat_likelihoods'], 
        #                     ]
        # hyper_likelihoods = [result1['hyper_likelihoods'], 
        #                     result2['hyper_likelihoods'], 
        #                     result3['hyper_likelihoods'], 
        #                     ]

        return {
            'img_hat': img_hat,
            'feat_likelihoods': feat_likelihoods,
            'hyper_likelihoods': hyper_likelihoods
        }

    def aux_loss(self):
        return self.codec_1.aux_loss() + self.codec_3.aux_loss()

    def update(self):
        self.codec_1.update()
        self.codec_3.update()
    
    def compress(self, img: torch.Tensor):
        result1 = self.codec_1.compress(img[0])
        result3 = self.codec_3.compress(img[2])

        return result1, result3
    
    def decompress(self, string1, shape1, string2, shape2):
        result1 = self.codec_1.decompress(string1, shape1)
        recon1 = self.prmodel1(result1['img_hat'])
        result3 = self.codec_3.decompress(string2, shape2)

        img_hat = [recon1, result1['img_hat'], result3['img_hat']]

        return {
            'img_hat': img_hat,
            'feat_likelihoods': {torch.tensor(0)},
            'hyper_likelihoods': {torch.tensor(0)}
        }
    
class prmodel(nn.Module):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=64):
        super().__init__()

        self.g_a = nn.Sequential(
            conv3x3(64, N, stride=1),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, 64, stride=1),
            nn.LeakyReLU(0.1)
        )

        self.upsampling = subpel_conv3x3(128, 64, 2)

    def forward(self, x):

        y = self.upsampling(x)
        # y = F.interpolate(x, scale_factor=2, mode='bilinear')
        y1 = self.g_a(y)
        y2 = y1 + y

        return y2