import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.models.priors import ScaleHyperprior
from compressai.models.priors import JointAutoregressiveHierarchicalPriors
import torch.nn.functional as F

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
    GDN,
)


class Encoder(nn.Module):
    def __init__(self, M = 192):
        super(Encoder, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=256)
        )
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=512)
        )
        self.bottom = nn.Conv2d(in_channels=1024, out_channels=M, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
    def forward(self, img):
        feats = self.head(img[0])
        feats = self.enc(torch.cat([img[1], feats], dim=1))
        feats = self.bottom(torch.cat([img[2],feats], dim=1))
        return feats
    

class Decoder(nn.Module):
    def __init__(self, M = 192):
        super(Decoder, self).__init__()

        self.lR1 = nn.LeakyReLU(0.1)

        self.out1_bn_leakyrelu = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1)
        ),
            nn.Sequential(
                nn.LeakyReLU(0.1)
        ),
            nn.Sequential(
                nn.LeakyReLU(0.1)
        )])

        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=M, out_channels=1024, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=1024, inverse=True)
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=512, inverse=True)
        )

        self.bottom = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))

    def forward(self, feats_hat):

        output_2 = self.head(feats_hat)
        output_2, feats_hat = torch.split(output_2, [512, 512], dim=1)
        output_1 = self.dec(feats_hat)
        output_1, feats_hat = torch.split(output_1, [256, 256], dim=1)
        output_0 = self.bottom(feats_hat)

        features = [output_0, output_1, output_2]
        features_clamp = [self.out1_bn_leakyrelu[i](features[i]) for i in range(len(features))] 
        return features_clamp


class DecoderEn(nn.Module):
    def __init__(self, M = 192):
        super(DecoderEn, self).__init__()

        self.lR1 = nn.LeakyReLU(0.1)

        self.out1_bn_leakyrelu = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1)
        ),
            nn.Sequential(
                nn.LeakyReLU(0.1)
        ),
            nn.Sequential(
                nn.LeakyReLU(0.1)
        )])

        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=M, out_channels=1024, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=1024, inverse=True)
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=512, inverse=True)
        )

        self.bottom = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))

    def forward(self, feats_hat):

        output_2 = self.head(feats_hat)
        _, feats_hat_2 = torch.split(output_2, [512, 512], dim=1)
        output_1 = self.dec(output_2)
        _, feats_hat_1 = torch.split(output_1, [256, 256], dim=1)
        feats_hat_0 = self.bottom(output_1)

        features = [feats_hat_0, feats_hat_1, feats_hat_2]
        features_clamp = [self.out1_bn_leakyrelu[i](features[i]) for i in range(len(features))] 
        return features_clamp
    

class HyperEncoder(nn.Module):
    def __init__(self, M=192):
        super(HyperEncoder, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.enc = nn.Sequential(
                nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(5, 5), stride=(2, 2), padding=2),
                nn.LeakyReLU(inplace=True),
        )

        self.bottom = nn.Conv2d(in_channels=M, out_channels=M, kernel_size=(5, 5), stride=(2, 2), padding=2)
    
    def forward(self, feats: torch.Tensor):
        feats = self.head(feats)
        feats = self.enc(feats)
        hyper = self.bottom(feats)
        return hyper

class HyperDecoder(nn.Module):
    def __init__(self, M=192):
        super(HyperDecoder, self).__init__()
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=M, out_channels=M, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.dec = nn.Sequential(
                nn.ConvTranspose2d(in_channels=M, out_channels=M * 3 // 2, kernel_size=(5, 5),
                                   stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
                nn.LeakyReLU(inplace=True)
        )

        self.bottom = nn.Conv2d(in_channels=M * 3 // 2, out_channels= M * 2, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1)

    def forward(self, hyper: torch.Tensor):

        hyper = self.head(hyper)
        hyper = self.dec(hyper)
        hyper_ctx_l = self.bottom(hyper)
        return hyper_ctx_l
    
class CoEncoder(nn.Module):
    def __init__(self):
        super(CoEncoder, self).__init__()

        self.head = ConvLstmsEnc(in_channels=128, out_channels=256)
        self.gdn1 = GDN(in_channels=256)

        self.enc = ConvLstmsEnc(in_channels=512, out_channels=512)
        self.gdn2 = GDN(in_channels=512)

        self.bottom = ConvLstmsEnc(in_channels=1024, out_channels=192)

    def forward(self, img, tEncCtx):
        
        if tEncCtx == None: tEncCtx = [None, None, None]

        c0, tEncCtx[0] = self.head(img[0], tEncCtx[0])
        c0 = self.gdn1(c0)

        c0_1, tEncCtx[1] = self.enc(torch.cat([img[1], c0], dim=1), tEncCtx[1])
        c0_1 = self.gdn2(c0_1)    

        c0_2, tEncCtx[2] = self.bottom(torch.cat([img[2], c0_1], dim=1), tEncCtx[2])
        return c0_2, tEncCtx
    
class CoDecoder(nn.Module):
    def __init__(self):
        super(CoDecoder, self).__init__()

        self.out1_bn_leakyrelu = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1)
        ),
            nn.Sequential(
                nn.LeakyReLU(0.1)
        ),
            nn.Sequential(
                nn.LeakyReLU(0.1)
        )])

        self.head = ConvLstmsDec(in_channels=192, out_channels=1024)
        self.dec = ConvLstmsDec(in_channels=512, out_channels=512)
        self.bottom = ConvLstmsDec(in_channels=256, out_channels=128)

    def forward(self, feats_hat, tDecCtx):

        if tDecCtx == None: tDecCtx = [None, None, None]
        output_2, tDecCtx[2] = self.head(feats_hat, tDecCtx[2])
        output_2, feats_hat = torch.split(output_2, [512, 512], dim=1)
        output_1, tDecCtx[1] = self.dec(feats_hat, tDecCtx[1])
        output_1, feats_hat = torch.split(output_1, [256, 256], dim=1)
        output_0, tDecCtx[0] = self.bottom(feats_hat, tDecCtx[0])

        features = [output_0, output_1, output_2]
        features_clamp = [self.out1_bn_leakyrelu[i](features[i]) for i in range(len(features))] 
        return features_clamp, tDecCtx

class ConvLstmsEnc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ConvLstmBlock_1 = NPUnit(in_channels, in_channels)
        self.ConvLstmBlock_2 = NPUnit(in_channels, in_channels)
        self.ConvLstmBlock_3 = NPUnit(in_channels, in_channels)
        self.EncUnit = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(2, 2), padding=2),
        )
    def forward(self, feat, tencctx):

        if tencctx == None:
            h, c = [], []
            zero_state = torch.zeros_like(feat) / 2.0
            zero_state.cuda()
            for i in range(3):
                h.append(zero_state)
                c.append(zero_state)
        else:
            h = tencctx[0]
            c = tencctx[1]

        h[0], c[0] = self.ConvLstmBlock_1(feat, h[0], c[0])
        h[1], c[1] = self.ConvLstmBlock_2(h[0], h[1], c[1])
        h[2], c[2] = self.ConvLstmBlock_3(h[1], h[2], c[2])
        # featEnh = torch.cat([feat, h[2]], dim=1)
        featEnh = feat - h[2]
        output = self.EncUnit(featEnh)

        return output, [h, c]# 这样可以吗？？？

class ConvLstmsDec(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.DecUnit = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=out_channels, inverse=True)
        )
        self.ConvLstmBlock_1 = NPUnit(out_channels, out_channels)
        self.ConvLstmBlock_2 = NPUnit(out_channels, out_channels)
        self.ConvLstmBlock_3 = NPUnit(out_channels, out_channels)

    def forward(self, feat, tdecctx):

        feat = self.DecUnit(feat)

        if tdecctx == None:
            h, c = [], []
            zero_state = torch.zeros_like(feat) / 2.0
            for i in range(3):
                h.append(zero_state)
                c.append(zero_state)
        else:
            h = tdecctx[0]
            c = tdecctx[1]

        h[0], c[0] = self.ConvLstmBlock_1(feat, h[0], c[0])
        h[1], c[1] = self.ConvLstmBlock_2(h[0], h[1], c[1])
        h[2], c[2] = self.ConvLstmBlock_3(h[1], h[2], c[2])
        output = feat + h[2]
        return output, [h, c] # 这样可以吗？？？


class NPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple = (3, 3)):
        super(NPUnit, self).__init__()
        same_padding = int((kernel_size[0]-1)/2)
        self.conv2d_x = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)
        self.conv2d_h = nn.Conv2d(in_channels=out_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)

    def forward(self, x, h, c):
        x_after_conv = self.conv2d_x(x)
        h_after_conv = self.conv2d_h(h)
        xi, xc, xf, xo = torch.chunk(x_after_conv, 4, dim=1)
        hi, hc, hf, ho = torch.chunk(h_after_conv, 4, dim=1)

        it = torch.sigmoid(xi+hi)
        ft = torch.sigmoid(xf+hf)
        new_c = (ft*c)+(it*torch.tanh(xc+hc))
        ot = torch.sigmoid(xo+ho)
        new_h = ot*torch.tanh(new_c)

        return new_h, new_c



class EncoderVCM(nn.Module):
    def __init__(self, M = 192):
        super(EncoderVCM, self).__init__()
        N = 128
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(128, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

    def forward(self, img):
        feats = self.g_a(img[0])
        return feats
    

class DecoderVCM(nn.Module):
    def __init__(self, M = 192):
        super(DecoderVCM, self).__init__()

        N = 128
        self.g_s_0 = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 128, 2),
            nn.LeakyReLU(0.1)
        )

        self.g_s_1 = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 256, 2),
            nn.LeakyReLU(0.1)
        )

        self.g_s_2 = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 512, 2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, feats_hat):

        output_0 = self.g_s_0(feats_hat)
        output_1 = self.g_s_1(feats_hat)
        output_2 = self.g_s_2(feats_hat)

        features = [output_0, output_1, output_2]
        return features
    

class EncoderBase1(nn.Module):
    def __init__(self, N, M):
        super(EncoderBase1, self).__init__()
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

    def forward(self, img):
        feats = self.g_a(img)
        return feats

class EncoderBase2(EncoderBase1):
    def __init__(self, N, M):
        super(EncoderBase2, self).__init__(N = N, M = M)
        N = 128
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(256, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

class EncoderBase3(EncoderBase1):
    def __init__(self, N, M):
        super(EncoderBase3, self).__init__(N = N, M = M)

        self.g_a = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2),
        )

class DecoderBase1(nn.Module):
    def __init__(self, N, M):
        super(DecoderBase1, self).__init__()

        self.g_s = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, feats_hat):
        output_0 = self.g_s(feats_hat)
        return output_0

class DecoderBase2(DecoderBase1):
    def __init__(self, N, M):
        super(DecoderBase2, self).__init__(N = N, M = M)

        self.g_s = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, feats_hat):
        output_0 = self.g_s(feats_hat)
        return output_0

class DecoderBase3(DecoderBase1):
    def __init__(self, N, M):
        super(DecoderBase3, self).__init__(N = N, M = M)

        self.g_s = nn.Sequential(
            ResidualBlock(M, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(0.1)
        )


