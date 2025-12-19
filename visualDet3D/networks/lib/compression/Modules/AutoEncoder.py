import torch
import torch.nn as nn
from compressai.layers import GDN
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import torch.nn.functional as F

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from visualDet3D.networks.lib.compression.Modules.Interaction import InteractionModule


count = 0
slice_height = 8
slice_width = 16
top_k = slice_height * slice_width // 2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # self.color_maps = {
        #     'viridis': cm.viridis,
        #     'plasma': cm.plasma,
        #     'inferno': cm.inferno,
        #     'magma': cm.magma,
        #     'cividis': cm.cividis,
        #     'jet': cm.jet,
        #     'rainbow': cm.rainbow,
        #     'coolwarm': cm.coolwarm
        # }
        
        # # 创建一个自定义的蓝-绿-红颜色映射
        # colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # 蓝、绿、红
        # self.color_maps['custom_rgb'] = LinearSegmentedColormap.from_list('custom_rgb', colors, N=100)


        # left_view encoder unit
        self.head_l = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128)
        )
        self.enc_l = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=256)
        )
        self.bottom_l = nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
        # right_view encoder unit
        self.head_r = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128)
        )
        self.enc_r = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=256)
        )
        self.bottom_r = nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)

        # interaction module
        self.interaction_blocks_0 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_1 = InteractionModule(channels=256, head_count=8)
        self.interaction_blocks_2 = InteractionModule(channels=192, head_count=8)

        # self.count = 0

    def forward(self, img_l, img_r):

        feats_l = self.head_l(img_l[0])
        feats_r = self.head_r(img_r[0])
        feats_l, feats_r = self.interaction_blocks_0(feats_l, feats_r)

        feats_l = self.enc_l(torch.cat([img_l[1], feats_l], dim=1))
        feats_r = self.enc_r(torch.cat([img_r[1], feats_r], dim=1))
        feats_l, feats_r = self.interaction_blocks_1(feats_l, feats_r)

        feats_l = self.bottom_l(torch.cat([img_l[2],feats_l], dim=1))
        feats_r = self.bottom_r(torch.cat([img_r[2],feats_r], dim=1))
        feats_l, feats_r = self.interaction_blocks_2(feats_l, feats_r)

        # self.write_torch_frame_grey(feats_l, path="visualizations_1/grey/1_4/left_{:4d}.png".format(self.count))
        # self.write_torch_frame_grey(feats_r, path="visualizations_1/grey/1_4/right_{:4d}.png".format(self.count))

        # print("successful save grey image {:4d}".format(self.count))
        # self.visualize_multi_channel(feats_l, save_path="visualizations_1/grey/1_16/left_{:4d}.png".format(self.count))
        # print("successfully visualize {:4d}-th left features".format(self.count))
        # self.visualize_multi_channel(feats_r, save_path="visualizations_1/grey/1_16/right_{:4d}.png".format(self.count))
        
        # print("successfully visualize {:4d}-th right features".format(self.count))

        # self.count = self.count + 1

        return feats_l, feats_r

    # def write_torch_frame_grey(self, frame, path):
    #     if not os.path.exists(os.path.dirname(os.path.abspath(path))):
    #         os.makedirs(os.path.dirname(os.path.abspath(path)))

    #     frame = torch.squeeze(frame, dim=0)
    #     frame_result = np.mean(frame.clone().cpu().detach().numpy(),axis=0)
    #     # min = np.mean(frame_result)
    #     # max = np.std(frame_result)
    #     min = np.min(frame_result)
    #     max = np.max(frame_result)
    #     frame_result = (frame_result - min)/(max-min) * 255
    #     frame_result = np.clip(np.rint(frame_result), 0, 255)
    #     frame_result = Image.fromarray(frame_result.astype('uint8')).convert('L')
    #     frame_result.save(path)
    
    def normalize(self, feature_map):
        """归一化特征图到[0, 1]范围"""
        min_val = np.min(feature_map)
        max_val = np.max(feature_map)
        if max_val - min_val < 1e-8:  # 避免除以零
            return np.zeros_like(feature_map)
        return (feature_map - min_val) / (max_val - min_val)
    
    def visualize_multi_channel(self, feature_map, cmap='viridis', title='Feature Map', save_path=None):
        """
        可视化多通道特征图
        
        参数:
            feature_map: 多通道特征图，形状为(C, H, W)或(H, W, C)
            num_channels: 要显示的通道数
            cmap: 颜色映射名称
            title: 图像标题
            save_path: 保存路径，为None则不保存
        """
        # 检查输入形状并调整为(C, H, W)
        feature_map = torch.squeeze(feature_map, dim=0)
        feature_map = feature_map.clone().cpu().detach().numpy()

        averaged_map = np.mean(feature_map, axis=0)

        normalized = self.normalize(averaged_map)

        # 选择颜色映射
        cmap = self.color_maps.get(cmap, cm.get_cmap(cmap))
        
        # 绘制特征图
        plt.figure(figsize=(8, 6))
        plt.imshow(normalized, cmap=cmap)
        plt.colorbar(orientation = 'horizontal', pad = 0.05)
        # plt.title(title, font={'family':'Times New Roman', 'size':16})
        plt.axis('off')
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"图像已保存至: {save_path}")
        
        # plt.show()



class FeatureCombine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p2Encoder_l = nn.Sequential(
            ResidualBlockWithStride(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.p2Encoder_r = nn.Sequential(
            ResidualBlockWithStride(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.p3Encoder_l = nn.Sequential(
            ResidualBlockWithStride(256, 256, stride=2),
            AttentionBlock(256),
            ResidualBlock(256, 256),
        )

        self.p3Encoder_r = nn.Sequential(
            ResidualBlockWithStride(256, 256, stride=2),
            AttentionBlock(256),
            ResidualBlock(256, 256),
        )

        self.p4Encoder_l = nn.Sequential(
            ResidualBlockWithStride(512, 192, stride=2),
            ResidualBlock(192, 192),
        )

        self.p4Encoder_r = nn.Sequential(
            ResidualBlockWithStride(512, 192, stride=2),
            ResidualBlock(192, 192),
        )

        # self.p5Encoder = nn.Sequential(
        #     conv3x3(N + M, M, stride=2),
        #     AttentionBlock(M),
        # )

    def forward(self, img_l, img_r):

        feats_l = self.p2Encoder_l(img_l[0])
        feats_r = self.p2Encoder_r(img_r[0])

        feats_l = self.p3Encoder_l(torch.cat([img_l[1], feats_l], dim=1))
        feats_r = self.p3Encoder_r(torch.cat([img_r[1], feats_r], dim=1))

        feats_l = self.p4Encoder_l(torch.cat([img_l[2],feats_l], dim=1))
        feats_r = self.p4Encoder_r(torch.cat([img_r[2],feats_r], dim=1))

        return feats_l, feats_r

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()

#         # left_view encoder unit
#         self.head_l = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
#             GDN(in_channels=128)
#         )
#         self.enc_l = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
#             GDN(in_channels=256)
#         )
#         # self.bottom_l = nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)

#         self.bottom_l = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2),
#             GDN(in_channels=192)
#         )

#         # right_view encoder unit
#         self.head_r = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
#             GDN(in_channels=128)
#         )
#         self.enc_r = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
#             GDN(in_channels=256)
#         )

#         self.bottom_r = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2),
#             GDN(in_channels=192)
#         )

#         # interaction module
#         self.interaction_blocks_0 = InteractionModule(channels=128, head_count=8)
#         self.interaction_blocks_1 = InteractionModule(channels=256, head_count=8)
#         self.interaction_blocks_2 = InteractionModule(channels=192, head_count=8)

#     def forward(self, img_l, img_r):

#         feats_l = self.head_l(img_l[0])
#         feats_r = self.head_r(img_r[0])
#         feats_l, feats_r = self.interaction_blocks_0(feats_l, feats_r)

#         feats_l = self.enc_l(torch.cat([img_l[1], feats_l], dim=1))
#         feats_r = self.enc_r(torch.cat([img_r[1], feats_r], dim=1))
#         feats_l, feats_r = self.interaction_blocks_1(feats_l, feats_r)

#         feats_l = self.bottom_l(torch.cat([img_l[2],feats_l], dim=1))
#         feats_r = self.bottom_r(torch.cat([img_r[2],feats_r], dim=1))
#         feats_l, feats_r = self.interaction_blocks_2(feats_l, feats_r)

#         return feats_l, feats_r

########################################  Decoder 3 ############################################
#Decoder 3: Decoder 1 add_output_l_2
class Decoder(nn.Module): # decoder 还没写
    def __init__(self):
        super(Decoder, self).__init__()

        self.head_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=512, inverse=True)
        )

        self.head_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=512, inverse=True)
        )

        self.dec_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=256, inverse=True)
        )

        self.dec_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=256, inverse=True)
        )

        self.bottom_l = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))
        self.bottom_r = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))

        self.interaction_blocks_0 = InteractionModule(channels=256, head_count=8)
        self.interaction_blocks_1 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_2 = InteractionModule(channels=192, head_count=8)

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor):


        feats_hat_l, feats_hat_r = self.interaction_blocks_2(feats_hat_l, feats_hat_r)
        output_l_2 = self.head_l(feats_hat_l)
        output_r_2 = self.head_r(feats_hat_r)  

        output_l_2, f_l_c = torch.split(output_l_2, [256, 256], dim=1)
        output_r_2, f_r_c = torch.split(output_r_2, [256, 256], dim=1) 
        feats_hat_l, feats_hat_r = self.interaction_blocks_0(f_l_c, f_r_c)
        output_l_1 = self.dec_l(torch.cat([output_l_2, f_l_c, feats_hat_l], dim=1))
        output_r_1 = self.dec_r(torch.cat([output_r_2, f_r_c, feats_hat_r], dim=1))

        output_l_1, f_l_c = torch.split(output_l_1, [128, 128], dim=1)
        output_r_1, f_r_c = torch.split(output_r_1, [128, 128], dim=1)
        feats_hat_l, feats_hat_r = self.interaction_blocks_1(f_l_c, f_r_c)
        output_l_0 = self.bottom_l(torch.cat([output_l_1, f_l_c, feats_hat_l], dim=1))
        output_r_0 = self.bottom_r(torch.cat([output_r_1, f_r_c, feats_hat_r], dim=1))

        features_l = [output_l_0, output_l_1, output_l_2]
        features_r = [output_r_0, output_r_1, output_r_2]

        return features_l, features_r

########################################  Decoder 3 revised ############################################
#Decoder 3: Decoder 1 add_output_l_2
# class Decoder(nn.Module): # decoder 还没写
#     def __init__(self):
#         super(Decoder, self).__init__()

#         self.head_l = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
#                                output_padding=(1, 1), padding=(2, 2)),
#             GDN(in_channels=512, inverse=True)
#         )

#         self.head_r = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
#                                output_padding=(1, 1), padding=(2, 2)),
#             GDN(in_channels=512, inverse=True)
#         )

#         self.dec_l = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
#                                 output_padding=(1, 1), padding=(2, 2)),
#             GDN(in_channels=256, inverse=True)
#         )

#         self.dec_r = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
#                                 output_padding=(1, 1), padding=(2, 2)),
#             GDN(in_channels=256, inverse=True)
#         )

#         self.bottom_l = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
#                                            output_padding=(1, 1), padding=(2, 2))
#         self.bottom_r = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
#                                            output_padding=(1, 1), padding=(2, 2))

#         self.interaction_blocks_0 = InteractionModule(channels=192, head_count=8)
#         self.interaction_blocks_1 = InteractionModule(channels=256, head_count=8)
#         self.interaction_blocks_2 = InteractionModule(channels=128, head_count=8)

#     def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor):

#         feats_hat_l, feats_hat_r = self.interaction_blocks_0(feats_hat_l, feats_hat_r)

#         output_l_2 = self.head_l(feats_hat_l)
#         output_r_2 = self.head_r(feats_hat_r)  

#         output_l_2, f_l_c = torch.split(output_l_2, [256, 256], dim=1)
#         output_r_2, f_r_c = torch.split(output_r_2, [256, 256], dim=1) 
#         feats_hat_l, feats_hat_r = self.interaction_blocks_1(f_l_c, f_r_c)
#         output_l_1 = self.dec_l(torch.cat([output_l_2, f_l_c, feats_hat_l], dim=1))
#         output_r_1 = self.dec_r(torch.cat([output_r_2, f_r_c, feats_hat_r], dim=1))

#         output_l_1, f_l_c = torch.split(output_l_1, [128, 128], dim=1)
#         output_r_1, f_r_c = torch.split(output_r_1, [128, 128], dim=1)
#         feats_hat_l, feats_hat_r = self.interaction_blocks_2(f_l_c, f_r_c)
#         output_l_0 = self.bottom_l(torch.cat([output_l_1, f_l_c, feats_hat_l], dim=1))
#         output_r_0 = self.bottom_r(torch.cat([output_r_1, f_r_c, feats_hat_r], dim=1))

#         features_l = [output_l_0, output_l_1, output_l_2]
#         features_r = [output_r_0, output_r_1, output_r_2]

#         return features_l, features_r

class FeatureSynthesis(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, N) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(N * 3, N * 2, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low


        self.p4Decoder_l = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 256, 2),
        )

        self.p4Decoder_r = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 256, 2),
        )

        self.p3Decoder_l = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 192, 2),
            AttentionBlock(192),
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 128, 2),
        )

        self.p3Decoder_r = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 192, 2),
            AttentionBlock(192),
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 128, 2),
        )

        self.p2Decoder_l = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 192, 2),
            AttentionBlock(192),
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 192, 2),
            ResidualBlock(192, 192),
            subpel_conv3x3(192, 64, 2),
        )

        self.p2Decoder_r = nn.Sequential(
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 192, 2),
            AttentionBlock(192),
            ResidualBlock(192, 192),
            ResidualBlockUpsample(192, 192, 2),
            ResidualBlock(192, 192),
            subpel_conv3x3(192, 64, 2),
        )

        self.decoder_attention_l = AttentionBlock(192)
        self.decoder_attention_r = AttentionBlock(192)

        self.fmb23_l = FeatureMixingBlock(64)
        self.fmb23_r = FeatureMixingBlock(64)

        self.fmb34_l = FeatureMixingBlock(128)
        self.fmb34_r = FeatureMixingBlock(128)

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor):
        y_hat_l = self.decoder_attention_l(feats_hat_l)
        p2_l = self.p2Decoder_l(y_hat_l)
        p3_l = self.fmb23_l(p2_l, self.p3Decoder_l(y_hat_l))
        p4_l = self.fmb34_l(p3_l, self.p4Decoder_l(y_hat_l))

        y_hat_r = self.decoder_attention_r(feats_hat_r)
        p2_r = self.p2Decoder_r(y_hat_r)
        p3_r = self.fmb23_r(p2_r, self.p3Decoder_r(y_hat_r))
        p4_r = self.fmb34_r(p3_r, self.p4Decoder_r(y_hat_r))

        features_l = [p2_l, p3_l, p4_l]
        features_r = [p2_r, p3_r, p4_r]

        return features_l, features_r


class HyperEncoder(nn.Module):
    def __init__(self):
        super(HyperEncoder, self).__init__()
        self.head_l = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.head_r = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.enc_l = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
                nn.LeakyReLU(inplace=True),
            )
        ])
        self.enc_r = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
                nn.LeakyReLU(inplace=True),
            )
        ])

        self.bottom_l = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.bottom_r = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2)

    def forward(self, feats_l: torch.Tensor, feats_r: torch.Tensor):
        feats_l = self.head_l(feats_l)
        feats_r = self.head_r(feats_r)

        feats_l = self.enc_l[0](feats_l)
        feats_r = self.enc_r[0](feats_r)

        hyper_l = self.bottom_l(feats_l)
        hyper_r = self.bottom_r(feats_r)
        return hyper_l, hyper_r
    
class HyperEncoder_Kim(nn.Module):
    def __init__(self):
        super(HyperEncoder_Kim, self).__init__()
        self.h_a_l = nn.Sequential(
            conv3x3(192, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128, stride=2),
        )

        self.h_a_r = nn.Sequential(
            conv3x3(192, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128),
            nn.LeakyReLU(inplace=True),
            conv3x3(128, 128, stride=2),
        )

    def forward(self, feats_l: torch.Tensor, feats_r: torch.Tensor):
        hyper_l = self.h_a_l(feats_l)
        hyper_r = self.h_a_r(feats_r)
        return hyper_l, hyper_r


class HyperDecoder(nn.Module):
    def __init__(self):
        super(HyperDecoder, self).__init__()
        self.head_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=192, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.head_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=192, kernel_size=(5, 5),
                               stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.dec_l = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=192, out_channels=288, kernel_size=(5, 5),
                                   stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
                nn.LeakyReLU(inplace=True)
            )
        ])

        self.dec_r = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=192, out_channels=288, kernel_size=(5, 5),
                                   stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
                nn.LeakyReLU(inplace=True)
            )
        ])

        self.bottom_l = nn.Conv2d(in_channels=288, out_channels=384, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1)
        self.bottom_r = nn.Conv2d(in_channels=288, out_channels=384, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1)

    def forward(self, hyper_l: torch.Tensor, hyper_r: torch.Tensor):
        hyper_l = self.head_l(hyper_l)
        hyper_r = self.head_r(hyper_r)

        hyper_l = self.dec_l[0](hyper_l)
        hyper_r = self.dec_r[0](hyper_r)

        hyper_ctx_l = self.bottom_l(hyper_l)
        hyper_ctx_r = self.bottom_r(hyper_r)
        return hyper_ctx_l, hyper_ctx_r
    

class HyperDecoder_Kim(nn.Module):
    def __init__(self):
        super(HyperDecoder_Kim, self).__init__()
        self.h_s_l = nn.Sequential(
            conv3x3(128, 192),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(192, 192, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(192, 288),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(288, 288, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(288, 384),
        )

        self.h_s_r = nn.Sequential(
            conv3x3(128, 192),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(192, 192, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(192, 288),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(288, 288, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(288, 384),
        )

    def forward(self, hyper_l: torch.Tensor, hyper_r: torch.Tensor):
        hyper_ctx_l = self.h_s_l(hyper_l)
        hyper_ctx_r = self.h_s_r(hyper_r)
        return hyper_ctx_l, hyper_ctx_r



############################################# Ablation Part ################################## 
class Encoder_Ablation(nn.Module):
    def __init__(self):
        super(Encoder_Ablation, self).__init__()

        self.head_l = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128)
        )

        self.head_r = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128)
        )

        self.enc_l = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
                GDN(in_channels=128)
            ) for _ in range(2)
        ])

        self.enc_r = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
                GDN(in_channels=128)
            ) for _ in range(2)
        ])

        self.bottom_l = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)

        self.bottom_r = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)

        self.interaction_blocks_0 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_1 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_2 = InteractionModule(channels=128, head_count=8)

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):
        feats_l = self.head_l(img_l)
        feats_r = self.head_r(img_r)
        feats_l, feats_r = self.interaction_blocks_0(feats_l, feats_r)

        feats_l = self.enc_l[0](feats_l)
        feats_r = self.enc_r[0](feats_r)
        feats_l, feats_r = self.interaction_blocks_1(feats_l, feats_r)

        feats_l = self.enc_l[1](feats_l)
        feats_r = self.enc_r[1](feats_r)
        feats_l, feats_r = self.interaction_blocks_2(feats_l, feats_r)

        feats_l = self.bottom_l(feats_l)
        feats_r = self.bottom_r(feats_r)

        return feats_l, feats_r


class Decoder_Ablation(nn.Module):
    def __init__(self):
        super(Decoder_Ablation, self).__init__()

        self.head_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=128, inverse=True)
        )

        self.head_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=128, inverse=True)
        )

        self.dec_l = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                   output_padding=(1, 1), padding=(2, 2)),
                GDN(in_channels=128, inverse=True),
            ) for _ in range(2)
        ])

        self.dec_r = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                   output_padding=(1, 1), padding=(2, 2)),
                GDN(in_channels=128, inverse=True),
            ) for _ in range(2)
        ])

        self.bottom_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=64, inverse=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=32, inverse=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
        )           
            
        self.bottom_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=64, inverse=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=32, inverse=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
        )    
                                           
        self.interaction_blocks_0 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_1 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_2 = InteractionModule(channels=128, head_count=8)

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor):
        feats_hat_l = self.head_l(feats_hat_l)
        feats_hat_r = self.head_r(feats_hat_r)

        feats_hat_l, feats_hat_r = self.interaction_blocks_2(feats_hat_l, feats_hat_r)
        feats_hat_l = self.dec_l[0](feats_hat_l)
        feats_hat_r = self.dec_r[0](feats_hat_r)

        feats_hat_l, feats_hat_r = self.interaction_blocks_1(feats_hat_l, feats_hat_r)
        feats_hat_l = self.dec_l[1](feats_hat_l)
        feats_hat_r = self.dec_r[1](feats_hat_r)

        feats_hat_l, feats_hat_r = self.interaction_blocks_0(feats_hat_l, feats_hat_r)
        img_hat_l = self.bottom_l(feats_hat_l)
        img_hat_r = self.bottom_r(feats_hat_r)

        return img_hat_l, img_hat_r

############################################# Ablation Part 2################################## 

class Encoder_Ablation_2(nn.Module):
    def __init__(self):
        super(Encoder_Ablation_2, self).__init__()

        # left_view encoder unit
        self.head_l = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
        )

        self.enc_l = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
        )

        self.bottom_l = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
        # right_view encoder unit
        self.head_r = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
        )

        self.enc_r = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2),
        )

        self.bottom_r = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2)

        # interaction module
        # self.interaction_blocks_0 = InteractionModule(channels=128, head_count=8)
        # self.interaction_blocks_1 = InteractionModule(channels=256, head_count=8)

    def forward(self, img_l, img_r):

        feats_l_1 = self.head_l(img_l[0])
        feats_r_1 = self.head_r(img_r[0])

        feats_l_2 = self.enc_l(img_l[1])
        feats_r_2 = self.enc_r(img_r[1])

        feats_l_3 = self.bottom_l(img_l[2])
        feats_r_3 = self.bottom_r(img_r[2])

        feats_l = torch.cat([feats_l_1,feats_l_2,feats_l_3], dim=1)
        feats_r = torch.cat([feats_r_1,feats_r_2,feats_r_3], dim=1)

        return feats_l, feats_r


class Decoder_Ablation_2(nn.Module): 
    def __init__(self):
        super(Decoder_Ablation_2, self).__init__()

        self.head_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=64, inverse=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=64, inverse=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
        )

        self.head_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=64, inverse=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=64, inverse=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
        )

        self.dec_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2))
        )
            

        self.dec_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=128, inverse=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2))
        )


        self.bottom_l = nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))
        self.bottom_r = nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))

        # self.interaction_blocks_0 = InteractionModule(channels=256, head_count=8)
        # self.interaction_blocks_1 = InteractionModule(channels=128, head_count=8)

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor):

        output_l = torch.split(feats_hat_l, [64, 64, 64], dim=1)
        output_r = torch.split(feats_hat_r, [64, 64, 64], dim=1)       

        output_l_0 = self.head_l(output_l[0])
        output_r_0 = self.head_r(output_r[0])

        output_l_1 = self.dec_l(output_l[1])
        output_r_1 = self.dec_r(output_r[1])

        output_l_2 = self.bottom_l(output_l[2])
        output_r_2 = self.bottom_r(output_r[2])

        features_l = [output_l_0, output_l_1, output_l_2]
        features_r = [output_r_0, output_r_1, output_r_2]

        return features_l, features_r

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

class Encoder_Ablation_sfatten(nn.Module):
    def __init__(self):
        super(Encoder_Ablation_sfatten, self).__init__()

        # left_view encoder unit
        self.head_l = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128)
        )
        self.enc_l = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=256)
        )
        self.bottom_l = nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
        self.nonlocal_block_l1 = EfficientAttention(in_channels=128, key_channels=128, head_count=8, value_channels=128)
        self.nonlocal_block_l2 = EfficientAttention(in_channels=256, key_channels=256, head_count=8, value_channels=256)

        # right_view encoder unit
        self.head_r = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=128)
        )
        self.enc_r = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            GDN(in_channels=256)
        )
        self.bottom_r = nn.Conv2d(in_channels=512, out_channels=192, kernel_size=(5, 5), stride=(2, 2), padding=2)
        
        self.nonlocal_block_r1 = EfficientAttention(in_channels=128, key_channels=128, head_count=8, value_channels=128)
        self.nonlocal_block_r2 = EfficientAttention(in_channels=256, key_channels=256, head_count=8, value_channels=256)

        # interaction module
        self.interaction_blocks_0 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_1 = InteractionModule(channels=256, head_count=8)
        self.interaction_blocks_2 = InteractionModule(channels=192, head_count=8)

    def forward(self, img_l, img_r):

        feats_l = self.head_l(img_l[0])
        feats_r = self.head_r(img_r[0])
        feats_l, feats_r = self.interaction_blocks_0(feats_l, feats_r)
        
        feats_sfl = self.nonlocal_block_l1(img_l[1], feats_l)
        feats_sfr = self.nonlocal_block_r1(img_r[1], feats_r)

        feats_l = self.enc_l(torch.cat([feats_sfl, feats_l], dim=1))
        feats_r = self.enc_r(torch.cat([feats_sfr, feats_r], dim=1))
        feats_l, feats_r = self.interaction_blocks_1(feats_l, feats_r)

        feats_sfl = self.nonlocal_block_l2(img_l[2], feats_l)
        feats_sfr = self.nonlocal_block_r2(img_r[2], feats_r)

        feats_l = self.bottom_l(torch.cat([feats_sfl, feats_l], dim=1))
        feats_r = self.bottom_r(torch.cat([feats_sfr, feats_r], dim=1))
        feats_l, feats_r = self.interaction_blocks_2(feats_l, feats_r)

        return feats_l, feats_r

class Decoder_Ablation_sfatten(nn.Module): # decoder 还没写
    def __init__(self):
        super(Decoder_Ablation_sfatten, self).__init__()

        self.head_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=512, inverse=True)
        )

        self.head_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=(5, 5), stride=(2, 2),
                               output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=512, inverse=True)
        )

        self.dec_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=256, inverse=True)
        )

        self.dec_r = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=(5, 5), stride=(2, 2),
                                output_padding=(1, 1), padding=(2, 2)),
            GDN(in_channels=256, inverse=True)
        )

        self.bottom_l = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))
        self.bottom_r = nn.ConvTranspose2d(in_channels=384, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                           output_padding=(1, 1), padding=(2, 2))

        self.nonlocal_block_l1 = EfficientAttention(in_channels=128, key_channels=128, head_count=8, value_channels=128)
        self.nonlocal_block_l2 = EfficientAttention(in_channels=256, key_channels=256, head_count=8, value_channels=256)
        self.nonlocal_block_r1 = EfficientAttention(in_channels=128, key_channels=128, head_count=8, value_channels=128)
        self.nonlocal_block_r2 = EfficientAttention(in_channels=256, key_channels=256, head_count=8, value_channels=256)

        self.interaction_blocks_0 = InteractionModule(channels=256, head_count=8)
        self.interaction_blocks_1 = InteractionModule(channels=128, head_count=8)
        self.interaction_blocks_2 = InteractionModule(channels=192, head_count=8)

    def forward(self, feats_hat_l: torch.Tensor, feats_hat_r: torch.Tensor):


        feats_hat_l, feats_hat_r = self.interaction_blocks_2(feats_hat_l, feats_hat_r)
        output_l_2 = self.head_l(feats_hat_l)
        output_r_2 = self.head_r(feats_hat_r)  

        output_l_2, f_l_c = torch.split(output_l_2, [256, 256], dim=1)
        output_r_2, f_r_c = torch.split(output_r_2, [256, 256], dim=1) 

        output_l_2 = self.nonlocal_block_l2(output_l_2, f_l_c)
        output_r_2 = self.nonlocal_block_r2(output_r_2, f_r_c)

        feats_hat_l, feats_hat_r = self.interaction_blocks_0(f_l_c, f_r_c)
        output_l_1 = self.dec_l(torch.cat([output_l_2, f_l_c, feats_hat_l], dim=1))
        output_r_1 = self.dec_r(torch.cat([output_r_2, f_r_c, feats_hat_r], dim=1))

        output_l_1, f_l_c = torch.split(output_l_1, [128, 128], dim=1)
        output_r_1, f_r_c = torch.split(output_r_1, [128, 128], dim=1)

        output_l_1 = self.nonlocal_block_l1(output_l_1, f_l_c)
        output_l_1 = self.nonlocal_block_r1(output_r_1, f_r_c)
        
        feats_hat_l, feats_hat_r = self.interaction_blocks_1(f_l_c, f_r_c)
        output_l_0 = self.bottom_l(torch.cat([output_l_1, f_l_c, feats_hat_l], dim=1))
        output_r_0 = self.bottom_r(torch.cat([output_r_1, f_r_c, feats_hat_r], dim=1))

        features_l = [output_l_0, output_l_1, output_l_2]
        features_r = [output_r_0, output_r_1, output_r_2]

        return features_l, features_r


if __name__ == "__main__":
    from thop import profile

    a = Encoder()
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 3, 512, 512)
    flops, params = profile(a, (x, y), verbose=False)
    print(" Encoder|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))

    b = HyperEncoder()
    x = torch.randn(1, 192, 32, 32)
    y = torch.randn(1, 192, 32, 32)
    flops, params = profile(b, (x, y), verbose=False)
    print(" Hyper Encoder|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))

    c = Decoder()
    x = torch.randn(1, 192, 32, 32)
    y = torch.randn(1, 192, 32, 32)
    flops, params = profile(c, (x, y), verbose=False)
    print(" Decoder|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))

    d = HyperDecoder()
    x = torch.randn(1, 128, 8, 8)
    y = torch.randn(1, 128, 8, 8)
    flops, params = profile(d, (x, y), verbose=False)
    print(" Hyper Decoder|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))
