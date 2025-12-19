from doctest import OutputChecker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.lib.ghost_module import ResGhostModule, GhostModule
from visualDet3D.networks.lib.PSM_cost_volume import PSMCosineModule, CostVolume
from visualDet3D.networks.backbones import resnet
from visualDet3D.networks.backbones.resnet import BasicBlock
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.lib.compression.Codec import Codec, Codec_Ablation, Codec_Ablation_2, Codec_MFCNet, Codec_Ablation_sfatten
from PIL import Image

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import os

class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4  = depth_channel_4 # 24
        self.depth_channel_8  = depth_channel_8 # 24
        self.depth_channel_16 = depth_channel_16 # 96

        input_features = depth_channel_4 # 24
        self.four_to_eight = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
            BasicBlock(3 * input_features, 3 * input_features),
        )
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
            #nn.Conv2d(3 * input_features, 3 * input_features, 3, padding=1, bias=False),
            #nn.BatchNorm2d(3 * input_features),
            #nn.ReLU(),
        )
        self.output_channel_num = 3 * input_features #1152

        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 96, 1),
        )


    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4_8 = self.four_to_eight(psv_volume_4)
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1)
        psv_16 = self.depth_reason(psv_volume_16)
        if self.training:
            return psv_16, self.depth_output(psv_16)
        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]])

class StereoMerging(nn.Module):
    def __init__(self, base_features):
        super(StereoMerging, self).__init__()
        self.cost_volume_0 = PSMCosineModule(downsample_scale=4, max_disp=96, input_features=base_features)
        PSV_depth_0 = self.cost_volume_0.depth_channel

        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=192, input_features=base_features * 2)
        PSV_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=192, input_features=base_features * 4, PSM_features=8)
        PSV_depth_2 = self.cost_volume_2.output_channel

        self.depth_reasoning = CostVolumePyramid(PSV_depth_0, PSV_depth_1, PSV_depth_2)
        self.final_channel = self.depth_reasoning.output_channel_num + base_features * 4

    def forward(self, left_x, right_x):
        PSVolume_0 = self.cost_volume_0(left_x[0], right_x[0])
        PSVolume_1 = self.cost_volume_1(left_x[1], right_x[1])
        PSVolume_2 = self.cost_volume_2(left_x[2], right_x[2])
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2) # c = 1152
        features = torch.cat([left_x[2], PSV_features], dim=1) # c = 1152 + 256 = 1408
        return features, depth_output

class YoloStereo3DCore(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore, self).__init__()
        self.backbone =resnet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging(base_features)


    def forward(self, images):

        Begin_time = time.time()
        print ("Starting to Encoding and Decoding time: {}",format(Begin_time))
        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]

        images = torch.cat([left_images, right_images], dim=0)
        features = self.backbone(images)

        Encoding_time = time.time()
        # print ("Encoding time: {}",format(Encoding_time - Begin_time))

        left_features  = [feature[0:batch_size] for feature in features]
        right_features = [feature[batch_size:]  for feature in features]

        features, depth_output = self.neck(left_features, right_features)

        output_dict = dict(features=features, depth_output=depth_output)


        Decoding_time = time.time()
        # print ("Decoding time: {}",format(Decoding_time-Encoding_time))
        
        return output_dict

# added jdc 加入压缩左右视图特征模块
class YoloStereo3DCore_VCM(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore_VCM, self).__init__()

        self.color_maps = {
            'viridis': cm.viridis,
            'plasma': cm.plasma,
            'inferno': cm.inferno,
            'magma': cm.magma,
            'cividis': cm.cividis,
            'jet': cm.jet,
            'rainbow': cm.rainbow,
            'coolwarm': cm.coolwarm
        }
        
        # 创建一个自定义的蓝-绿-红颜色映射
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # 蓝、绿、红
        self.color_maps['custom_rgb'] = LinearSegmentedColormap.from_list('custom_rgb', colors, N=100)

        self.backbone =resnet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging(base_features)

        self.stereo_compression = Codec()
        self.reconstruction = Reconstruction()
        self.count = 0

    def forward(self, images):

        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]

        images = torch.cat([left_images, right_images], dim=0) # [4, 3, 288, 1280]

        features = self.backbone(images)
        

        left_features  = [feature[0:batch_size] for feature in features] # [[4, 64, 72, 320], [4, 128, 36, 160], [4, 256, 18, 80]]
        right_features = [feature[batch_size:]  for feature in features] 

        # # features visualization
        # self.write_torch_frame_grey(left_features[0], path='0_left.png')
        # self.write_torch_frame_grey(left_features[1], path="1_left.png")
        # self.write_torch_frame_grey(left_features[2], path="2_left.png")

        # self.write_torch_frame_grey(right_features[0], path="0_right.png")
        # self.write_torch_frame_grey(right_features[1], path="1_right.png")
        # self.write_torch_frame_grey(right_features[2], path="2_right.png")


        size_base = left_features[0].shape[2:]
        size = size_base
        size_pad = None
        left_features_pad = []
        right_features_pad = []
        # padding features for compression 

        for idx in range(len(left_features)-1, -1, -1):
            left_feat_pad = get_pad_result(left_features[idx], size=size_pad)
            right_feat_pad = get_pad_result(right_features[idx], size=size_pad)
            left_features_pad.append(left_feat_pad)
            right_features_pad.append(right_feat_pad)
            size_pad = torch.as_tensor(left_feat_pad.shape[2:]) * 2

        left_features_pad_ = []
        right_features_pad_ = []
        for left_feat, right_feat in zip(reversed(left_features_pad), reversed(right_features_pad)):
            left_features_pad_.append(left_feat)
            right_features_pad_.append(right_feat)

        # added jdc
        result = self.stereo_compression(left_features_pad_, right_features_pad_) # 这里计算bpp和psnr的时候需要注意相应修改

        left_features_unpad = []
        right_features_unpad = []
        # unpadding features for neck
        for left_feature, right_feature in zip(result['img_hat'][0], result['img_hat'][1]):
            left_features_unpad.append(get_unpad_result(left_feature, size))
            right_features_unpad.append(get_unpad_result(right_feature, size))
            size = torch.tensor(torch.as_tensor(size) * 0.5, dtype=int)


        # # features visualization
        # self.write_torch_frame_grey(left_features_unpad[0], path="visualizations_1/grey/1_16_g/0_left_unpad_{:4d}.png".format(self.count))
        # self.write_torch_frame_grey(left_features_unpad[1], path="1_left_unpad.png")
        # self.write_torch_frame_grey(left_features_unpad[2], path="2_left_unpad.png")
        # self.visualize_multi_channel(left_features_unpad[0], save_path="visualizations_1/grey/1_256/left_{:4d}.png".format(self.count))
        # self.write_torch_frame_grey(right_features_unpad[0], path="visualizations_1/grey/1_16_g/0_right_unpad_{:4d}.png".format(self.count))
        # self.write_torch_frame_grey(right_features_unpad[1], path="1_right_unpad.png")
        # self.write_torch_frame_grey(right_features_unpad[2], path="2_right_unpad.png")
        # self.visualize_multi_channel(right_features_unpad[0], save_path="visualizations_1/grey/1_256/right_{:4d}.png".format(self.count))

        self.count = self.count + 1


        features, depth_output = self.neck(left_features_unpad, right_features_unpad)
        # features, depth_output = self.neck(left_features, right_features)
        
        #left_images_recon, right_images_recon = self.reconstruction(left_features_unpad, right_features_unpad)
        output_dict = dict(features=features, depth_output=depth_output, \
            result=result, left_images_recon=left_images, right_images_recon=right_images) # 返回likehood和hat

        # features, depth_output = self.neck(left_features, right_features) # replaced
        # output_dict = dict(features=features, depth_output=depth_output, result=result)
        return output_dict

    def encode_decode(self, images):

        Begin_time = time.time()
        print ("Starting to Encoding and Decoding time: {}",format(Begin_time))

        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]
        images = torch.cat([left_images, right_images], dim=0) # [4, 3, 288, 1280]
        features = self.backbone(images)
    
        left_features  = [feature[0:batch_size] for feature in features] # [[4, 64, 72, 320], [4, 128, 36, 160], [4, 256, 18, 80]]
        right_features = [feature[batch_size:]  for feature in features] 
        size_base = left_features[0].shape[2:]
        size = size_base
        size_pad = None
        left_features_pad = []
        right_features_pad = []
        # padding features for compression 

        for idx in range(len(left_features)-1, -1, -1):
            left_feat_pad = get_pad_result(left_features[idx], size=size_pad)
            right_feat_pad = get_pad_result(right_features[idx], size=size_pad)
            left_features_pad.append(left_feat_pad)
            right_features_pad.append(right_feat_pad)
            size_pad = torch.as_tensor(left_feat_pad.shape[2:]) * 2

        left_features_pad_ = []
        right_features_pad_ = []
        for left_feat, right_feat in zip(reversed(left_features_pad), reversed(right_features_pad)):
            left_features_pad_.append(left_feat)
            right_features_pad_.append(right_feat)

        # added jdc
        self.stereo_compression.update()
        encoded_results = self.stereo_compression.compress(left_features_pad_, right_features_pad_)

        Encoding_time = time.time()
        print ("Encoding time: {}",format(Encoding_time - Begin_time))

        decoded_results = self.stereo_compression.decompress(encoded_results["strings"], encoded_results["shape"])
        result = decoded_results

        left_features_unpad = []
        right_features_unpad = []
        # unpadding features for neck
        for left_feature, right_feature in zip(result['img_hat'][0], result['img_hat'][1]):
            left_features_unpad.append(get_unpad_result(left_feature, size))
            right_features_unpad.append(get_unpad_result(right_feature, size))
            size = torch.tensor(torch.as_tensor(size) * 0.5, dtype=int)

        features, depth_output = self.neck(left_features_unpad, right_features_unpad)
        # features, depth_output = self.neck(left_features, right_features)
        
        #left_images_recon, right_images_recon = self.reconstruction(left_features_unpad, right_features_unpad)
        output_dict = dict(features=features, depth_output=depth_output, \
            result=result, left_images_recon=left_images, right_images_recon=right_images) # 返回likehood和hat

        Decoding_time = time.time()
        print ("Decoding time: {}",format(Decoding_time-Encoding_time))

        # features, depth_output = self.neck(left_features, right_features) # replaced
        # output_dict = dict(features=features, depth_output=depth_output, result=result)
        return output_dict

    def write_torch_frame_grey(self, frame, path):
        if not os.path.exists(os.path.dirname(os.path.abspath(path))):
            os.makedirs(os.path.dirname(os.path.abspath(path)))

        frame = torch.squeeze(frame, dim=0)
        frame_result = np.mean(frame.clone().cpu().detach().numpy(),axis=0)
        # min = np.mean(frame_result)
        # max = np.std(frame_result)
        min = np.min(frame_result)
        max = np.max(frame_result)
        frame_result = (frame_result - min)/(max-min) * 255
        frame_result = np.clip(np.rint(frame_result), 0, 255)
        frame_result = Image.fromarray(frame_result.astype('uint8')).convert('L')
        frame_result.save(path)
    
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
            # print(f"图像已保存至: {save_path}")

class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()

        self.left_branch = RecontructionBranch()
        self.right_branch = RecontructionBranch()

    def forward(self, left_features, right_features):

        left_output = self.left_branch(left_features)
        right_output = self.right_branch(right_features)

        return left_output, right_output

class RecontructionBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.left_resblock_1 = ResBlockUpsampling(in_channels=256, out_channels=128)
        self.left_resblock_2 = ResBlockUpsampling(in_channels=128, out_channels=64)
        self.left_resblock_3 = ResBlockUpsampling(in_channels=64, out_channels=32)
        self.left_resblock_bottem = ResBlockUpsampling(in_channels=32, out_channels=3, activation=False)

    def forward(self, left_features):

        left_output = self.left_resblock_1(left_features[2]) + left_features[1]
        left_output = self.left_resblock_2(left_output) + left_features[0]
        left_output = self.left_resblock_bottem(self.left_resblock_3(left_output))
        #left_output = torch.clamp(left_output, min=0.0, max=1.0) # 这里需要看一下制作数据的时候，数据是不是在0-1之间

        return left_output

def get_pad_result(cur_frame, size=None):

    if not size == None:
        h, w = size[0:2]
        h = int(h - cur_frame.shape[2])
        w = int(w - cur_frame.shape[3])
        return F.pad(cur_frame, (0, w, 0, h), 'constant', 0)
    else:
        h = int(8 - cur_frame.shape[2] % 8) % 8
        w = int(8 - cur_frame.shape[3] % 8) % 8
        #print(cur_frame.shape)
        if h % 8 == 0 and w % 8 == 0: # no padding
            return cur_frame
        else:
            cur_frame = F.pad(cur_frame, (0, w, 0, h), 'constant', 0)
            # = F.pad(cur_frame, (w, 0, h, 0), 'constant', 0)
            #print(cur_frame.shape)
            return cur_frame

def get_unpad_result(cur_frame, size_base):
    h = size_base[0]
    w = size_base[1]
    return cur_frame[:, :, 0:h, 0:w]


class ResBlockUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, activation:bool=True):
        super().__init__()

        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), stride=1, padding=1),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(inplace=True),

        nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        )

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.activation = activation

    def forward(self, x: torch.Tensor):

        y = self.upsampling(x) + self.block(x)

        if self.activation:
            return self.LeakyReLU(y)
        else:
            return y


# added jdc 加入压缩左右视图特征模块
class YoloStereo3DCore_VCM_Ablation(nn.Module):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore_VCM_Ablation, self).__init__()
        self.backbone =resnet(**backbone_arguments)

        base_features = 256 if backbone_arguments['depth'] > 34 else 64
        self.neck = StereoMerging(base_features)

        self.stereo_compression = Codec_Ablation()
        self.reconstruction = Reconstruction()

    @staticmethod
    def save_feature_map(features):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = features[0].shape[0]
        width_list = [2, 2, 2]
        height_list = [32,64,128] # [4, 64, 72, 320], [4, 128, 36, 160], [4, 256, 18, 80]]

        tile_big = torch.empty((batch_size, features[0].shape[2] * height_list[0], 0)).to(device)

        for blk, width, height in zip(features, width_list, height_list):

            big_blk = torch.empty((batch_size, blk.shape[2] * height, 0)).to(device)

            for row in range(width):
                big_blk_col = torch.empty((batch_size, 0, blk.shape[3])).to(device)
                for col in range(height):
                    tile = blk[:, col + row * height, :, :]
                    big_blk_col = torch.hstack((big_blk_col, tile))
                big_blk = torch.dstack((big_blk, big_blk_col))
            tile_big =  torch.dstack((tile_big, big_blk))
        tile_big = torch.unsqueeze(tile_big, dim=1)
        return tile_big


    def feat2feat(self, png):

        vectors_width = png.shape[3]
        v2_w = int(vectors_width / 560 * 320)
        v3_w = int(vectors_width / 560 * 480)

        v2_blk = png[:,:,:,:v2_w]
        v3_blk = png[:,:,:,v2_w: v3_w]
        v4_blk = png[:,:,:,v3_w:vectors_width]

        feature_2 = self.feature_slice(v2_blk, [v2_blk.shape[2] // 32, v2_blk.shape[3]//2])
        feature_3 = self.feature_slice(v3_blk, [v3_blk.shape[2] // 64, v3_blk.shape[3]//2])
        feature_4 = self.feature_slice(v4_blk, [v4_blk.shape[2] // 128, v4_blk.shape[3]//2])

        return [feature_2,feature_3,feature_4]

    @staticmethod
    def feature_slice(image, shape):
        height = image.shape[2]
        width = image.shape[3]

        blk_height = shape[0]
        blk_width = shape[1]
        blk = []

        for x in range(width // blk_width):
            for y in range(height // blk_height):
                y_lower = y * blk_height
                y_upper = (y + 1) * blk_height
                x_lower = x * blk_width
                x_upper = (x + 1) * blk_width
                blk.append(image[:, :, y_lower:y_upper, x_lower:x_upper])
        feature = torch.cat(blk, dim=1)
        return feature


    def forward(self, images):

        batch_size = images.shape[0]
        left_images = images[:, 0:3, :, :]
        right_images = images[:, 3:, :, :]

        images = torch.cat([left_images, right_images], dim=0) # [4, 3, 288, 1280]

        features = self.backbone(images)

        left_features  = [feature[0:batch_size] for feature in features] # [[4, 64, 72, 320], [4, 128, 36, 160], [4, 256, 18, 80]], 12, 40
        right_features = [feature[batch_size:]  for feature in features] # 

        left_image = self.save_feature_map(left_features)
        right_image = self.save_feature_map(right_features)

        # added jdc
        shape = [2304,1280] # /64 [36, 20]
        left_image = get_pad_result(left_image, shape)
        right_image = get_pad_result(right_image, shape)
        result = self.stereo_compression(left_image, right_image) # 这里计算bpp和psnr的时候需要注意相应修改
        left_images_unpad = get_unpad_result(result['img_hat'][0], [2304,1120])
        right_images_unpad = get_unpad_result(result['img_hat'][1], [2304,1120])

        left_feature_compressed = self.feat2feat(left_images_unpad)
        right_feature_compressed = self.feat2feat(right_images_unpad)

        features, depth_output = self.neck(left_feature_compressed, right_feature_compressed)
        
        left_images_recon, right_images_recon = self.reconstruction(left_feature_compressed, right_feature_compressed)
        output_dict = dict(features=features, depth_output=depth_output, \
            result=result, left_images_recon=left_images_recon, right_images_recon=right_images_recon) # 返回likehood和hat

        return output_dict


# added jdc 加入压缩左右视图特征模块
class YoloStereo3DCore_VCM_Ablation_2(YoloStereo3DCore_VCM):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore_VCM_Ablation_2, self).__init__(backbone_arguments)

        self.stereo_compression = Codec_Ablation_2()

# added jdc 加入压缩左右视图特征模块
class YoloStereo3DCore_VCM_Ablation_sfatten(YoloStereo3DCore_VCM):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore_VCM_Ablation_sfatten, self).__init__(backbone_arguments)

        self.stereo_compression = Codec_Ablation_sfatten()

# added jdc 加入压缩左右视图特征模块
class YoloStereo3DCore_VCM_MFCNet(YoloStereo3DCore_VCM):
    """
        Inference Structure of YoloStereo3D
        Similar to YoloMono3D,
        Left and Right image are fed into the backbone in batch. So they will affect each other with BatchNorm2d.
    """
    def __init__(self, backbone_arguments):
        super(YoloStereo3DCore_VCM_MFCNet, self).__init__(backbone_arguments)

        self.stereo_compression = Codec_MFCNet()

def get_pad_result(cur_frame, size=None):

    if not size == None:
        h, w = size[0:2]
        h = int(h - cur_frame.shape[2])
        w = int(w - cur_frame.shape[3])
        return F.pad(cur_frame, (0, w, 0, h), 'constant', 0)
    else:
        h = int(8 - cur_frame.shape[2] % 8) % 8
        w = int(8 - cur_frame.shape[3] % 8) % 8
        #print(cur_frame.shape)
        if h % 8 == 0 and w % 8 == 0: # no padding
            return cur_frame
        else:
            cur_frame = F.pad(cur_frame, (0, w, 0, h), 'constant', 0)
            # = F.pad(cur_frame, (w, 0, h, 0), 'constant', 0)
            #print(cur_frame.shape)
            return cur_frame

def get_unpad_result(cur_frame, size_base):
    h = size_base[0]
    w = size_base[1]
    return cur_frame[:, :, 0:h, 0:w]