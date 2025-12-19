import torch
import torch.nn as nn
from compressai.layers import GDN, MaskedConv2d
from compressai.models import CompressionModel
from compressai.models.priors import ScaleHyperprior, JointAutoregressiveHierarchicalPriors
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
import torch.nn.functional as F
from compressai.models.utils import update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
from feature_compression_modules import *
import math
import warnings


# •	512 × 34×19
# •	256 × 68×38
# •	128 × 136×76
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min_: float = SCALES_MIN, max_: float = SCALES_MAX, levels: int = SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min_), math.log(max_), levels))

class CodecTest(CompressionModel):
    def __init__(self, N=192, M=192):
        super(CodecTest, self).__init__(entropy_bottleneck_channels=128)
        self.encoder0 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.1)
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.1)
        )

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()

        self.entropy_bottleneck = EntropyBottleneck(channels=192)

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

    def forward(self, img: torch.Tensor):

        output_0 = self.encoder0(img[0])
        output_1 = self.encoder1(img[1])
        output_2 = self.encoder2(img[2])

        img_hat = [output_0, output_1, output_2]

        hyper_likelihoods = torch.tensor(1) # 注意这里一定是设置为1而不是0
        feat_likelihoods = torch.tensor(1)

        return {
            'img_hat': img_hat,
            'feat_likelihoods': feat_likelihoods,
            'hyper_likelihoods': hyper_likelihoods
        }
    
# class CodecTest(CompressionModel):
#     def __init__(self, N=192, M=192):
#         super(CodecTest, self).__init__(entropy_bottleneck_channels=128)
        
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#         self.hyper_encoder = HyperEncoder()
#         self.hyper_decoder = HyperDecoder()

#         self.entropy_bottleneck = EntropyBottleneck(channels=192)

#         self.entropy_parameters = nn.Sequential(
#             nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
#         )
#         self.context_prediction = MaskedConv2d(
#             M, 2 * M, kernel_size=5, padding=2, stride=1
#         )
#         self.gaussian = GaussianConditional(None)

#     def forward(self, img: torch.Tensor):

#         feats = self.encoder(img)
#         hyper = self.hyper_encoder(feats)

#         hyper_hat, hyper_likelihoods = self.entropy_bottleneck(hyper)
#         hyper_ctx = self.hyper_decoder(hyper_hat)

#         feats_hat = self.gaussian.quantize(feats, "noise" if self.training else "dequantize")
#         ctx_params = self.context_prediction(feats_hat)
#         params = self.entropy_parameters(torch.cat((hyper_ctx, ctx_params), dim=1))

#         scales, means = params.chunk(2, 1)
#         _, feat_likelihoods = self.gaussian(feats, F.relu(scales), means=means)


#         hyper_likelihoods = torch.tensor(1) # 注意这里一定是设置为1而不是0
#         feat_likelihoods = torch.tensor(1)
#         img_hat = self.decoder(feats)
#         return {
#             'img_hat': img_hat,
#             'feat_likelihoods': feat_likelihoods,
#             'hyper_likelihoods': hyper_likelihoods
#         }

#     def load_state_dict(self, state_dict):
#         update_registered_buffers(
#             self.gaussian,
#             "gaussian",
#             ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
#             state_dict,
#         )
#         super().load_state_dict(state_dict)

#     def update(self, scale_table: torch.Tensor = None, force: bool = False):
#         if scale_table is None:
#             scale_table = get_scale_table()
#         updated = self.gaussian_l.update_scale_table(scale_table, force=force)
#         updated |= self.gaussian_r.update_scale_table(scale_table, force=force)
#         updated |= super().update(force=force)
#         return updated

class Codec(CompressionModel):
    def __init__(self, N=192, M=384):
        super(Codec, self).__init__(entropy_bottleneck_channels=128)

        self.encoder = Encoder(M)
        self.decoder = Decoder(M)

        self.hyper_encoder = HyperEncoder(M)
        self.hyper_decoder = HyperDecoder(M)

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
        updated = self.gaussian_l.update_scale_table(scale_table, force=force)
        updated |= self.gaussian_r.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
class CodecEnDec(Codec):
    def __init__(self, N=192, M=384):
        super(CodecEnDec, self).__init__(M)
        self.decoder = DecoderEn(M)

class CoCodecSic(CompressionModel):
    def __init__(self, N=192, M=192):
        super(CoCodecSic, self).__init__(entropy_bottleneck_channels=128)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()

        self.entropy_bottleneck = EntropyBottleneck(channels=192)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 18 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        self.gaussian = GaussianConditional(None)
        self.temporal_prediction = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=288,  kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=288, out_channels=384,  kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, img: torch.Tensor, tCtx: torch.tensor): #TODO 将上一帧的feats_hat 拿过来算ctx_temporal
        # tCtx = [tEncCtx, tDecCtx]
        # tEncCtx = [[h, c], [h, c]]
        # h = [h1, h2, h3]
        
        
        feats = self.encoder(img) # tCtxEncoder
        hyper = self.hyper_encoder(feats)

        if tCtx == None: tCtx = torch.zeros_like(feats) / 2.0
        feats = feats - tCtx # added

        hyper_hat, hyper_likelihoods = self.entropy_bottleneck(hyper)
        hyper_ctx = self.hyper_decoder(hyper_hat)

        feats_hat = self.gaussian.quantize(feats, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(feats_hat)
        tem_params = self.temporal_prediction(feats_hat) # added
        params = self.entropy_parameters(torch.cat((hyper_ctx, ctx_params, tem_params), dim=1)) # revised

        scales, means = params.chunk(2, 1)
        _, feat_likelihoods = self.gaussian(feats, F.relu(scales), means=means)
        feats_hat = feats_hat + tCtx

        img_hat = self.decoder(feats_hat) # tCtxDecoder

        return {
            'img_hat': img_hat,
            'feat_likelihoods': feat_likelihoods,
            'hyper_likelihoods': hyper_likelihoods,
            'tCtx': feats_hat
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
        updated = self.gaussian_l.update_scale_table(scale_table, force=force)
        updated |= self.gaussian_r.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

class CoCodec(CompressionModel):
    def __init__(self, N=192, M=192):
        super(CoCodec, self).__init__(entropy_bottleneck_channels=128)

        self.encoder = CoEncoder()
        self.decoder = CoDecoder()

        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()

        self.entropy_bottleneck = EntropyBottleneck(channels=192)

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


    def forward(self, img: torch.Tensor, tCtx: torch.tensor): #TODO 将上一帧的feats_hat 拿过来算ctx_temporal
        # tCtx = [tEncCtx, tDecCtx]
        # tEncCtx = [[h, c], [h, c]]
        # h = [h1, h2, h3]
        if tCtx == None: tCtx = [None, None]

        feats, tCtx[0] = self.encoder(img, tCtx[0]) # tCtxEncoder
        hyper = self.hyper_encoder(feats)

        hyper_hat, hyper_likelihoods = self.entropy_bottleneck(hyper)
        hyper_ctx = self.hyper_decoder(hyper_hat)

        feats_hat = self.gaussian.quantize(feats, "noise" if self.training else "dequantize")
        ctx_params = self.context_prediction(feats_hat)
        params = self.entropy_parameters(torch.cat((hyper_ctx, ctx_params), dim=1))

        scales, means = params.chunk(2, 1)
        _, feat_likelihoods = self.gaussian(feats, F.relu(scales), means=means)

        img_hat, tCtx[1] = self.decoder(feats_hat, tCtx[1]) # tCtxDecoder
        return {
            'img_hat': img_hat,
            'feat_likelihoods': feat_likelihoods,
            'hyper_likelihoods': hyper_likelihoods,
            'tCtx': tCtx
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
        updated = self.gaussian_l.update_scale_table(scale_table, force=force)
        updated |= self.gaussian_r.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    

class CodecVCM(CompressionModel):

    def __init__(self, N=192, M=192):
        super(CodecVCM, self).__init__(entropy_bottleneck_channels=128)

        self.encoder = EncoderVCM(M = M)
        self.decoder = DecoderVCM(M = M)

        self.hyper_encoder = HyperEncoder(M)
        self.hyper_decoder = HyperDecoder(M)

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