import sys
sys.path.append('/media/D/visualDet3D-master/')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import nms
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.utils.timer import profile
from visualDet3D.networks.heads import losses
from visualDet3D.networks.detectors.yolostereo3d_core import YoloStereo3DCore, \
YoloStereo3DCore_VCM, YoloStereo3DCore_VCM_Ablation, YoloStereo3DCore_VCM_Ablation_2, YoloStereo3DCore_VCM_MFCNet, YoloStereo3DCore_VCM_Ablation_sfatten
from visualDet3D.networks.heads.detection_3d_head import StereoHead
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.backbones.resnet import BasicBlock
from compressai.models.utils import update_registered_buffers



@DETECTOR_DICT.register_module
class Stereo3D(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )
        # 计算类别和回归框预测

        anchors = self.bbox_head.get_anchor(left_images, P2) # 根据初始参数cfg，计算真实框

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2) # 计算类别损失和回归框损失

        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None: # 如果存在深度图
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        
        return scores, bboxes, cls_indexes


    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)



# added jdc
@DETECTOR_DICT.register_module
class Stereo3D_VCM(nn.Module):
    """
        Stereo3D_VCM
    """
    def __init__(self, network_cfg, train_augmentation_cfg):
        super(Stereo3D_VCM, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

        self.train_augmentation_cfg = train_augmentation_cfg

        self.dist = nn.MSELoss().to('cuda')

    def dis(self, A, B):

        for T in self.train_augmentation_cfg:
            if T.type_name == 'Normalize':
                A = A.permute(0, 2, 3, 1)
                B = B.permute(0, 2, 3, 1)
                A = A * torch.as_tensor(T.keywords.stds).to('cuda') + torch.as_tensor(T.keywords.mean).to('cuda')
                B = B * torch.as_tensor(T.keywords.stds).to('cuda') + torch.as_tensor(T.keywords.mean).to('cuda')
                break

        A = torch.clamp(A, min=0, max=1)
        B = torch.clamp(B, min=0, max=1)

        A = A.permute(0, 3, 1, 2)
        B = B.permute(0, 3, 1, 2)
        result = self.dist(A, B) 
        return result, A, B

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore_VCM(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(torch.cat([left_images, right_images], dim=1)) # [4, 3, 288, 1280]
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )
        # 计算类别和回归框预测

        anchors = self.bbox_head.get_anchor(left_images, P2) # 根据初始参数cfg，计算真实框

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2) # 计算类别损失和回归框损失

        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None: # 如果存在深度图
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss # loss_dict['total']没有加上深度图loss added jdc
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)

        # 添加bpp loss 和 aux loss
        # calcualte bpp
        bpps = {}
        num_pixels = left_images.shape[0] * left_images.shape[2] * left_images.shape[3]
        bpp_feat_l, bpp_feat_r, bpp_hyper_l, bpp_hyper_r = 0, 0, 0, 0
        for likelihood in output_dict['result']['feat_likelihoods'][0]:
            bpp_feat_l = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_feat_l
        for likelihood in output_dict['result']['feat_likelihoods'][1]:
            bpp_feat_r = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_feat_r
        for likelihood in output_dict['result']['hyper_likelihoods'][0]:
            bpp_hyper_l = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_hyper_l
        for likelihood in output_dict['result']['hyper_likelihoods'][1]:
            bpp_hyper_r = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_hyper_r
        bpps['bpp_left'] = bpp_feat_l + bpp_hyper_l
        bpps['bpp_right'] = bpp_feat_r + bpp_hyper_r
        self.bpp_total = (bpp_feat_l + bpp_feat_r + bpp_hyper_l + bpp_hyper_r)

        # calculate distortion # 注意归一化计算的问题
        Pictures = {}
        dist_left, rec_left, ori_left = self.dis(output_dict['left_images_recon'], left_images)
        dist_right, rec_right, ori_right = self.dis(output_dict['right_images_recon'], right_images)
        Pictures['rec_left'] = rec_left
        Pictures['ori_left'] = ori_left
        Pictures['rec_right'] = rec_right
        Pictures['ori_right'] = ori_right
        self.dist_total = dist_left + dist_right

        #calculate rdo
        rd_loss = self.cal_rd_cost(distortion=self.dist_total, bpp=self.bpp_total, \
            lambda_weight=self.network_cfg.compression_args.lambda_weight)
        # rd_loss = self.cal_rd_cost(distortion=self.dist_total, bpp=0, \
        #     lambda_weight=self.network_cfg.compression_args.lambda_weight)

        aux_loss = self.core.stereo_compression.aux_loss()
        loss_dict['rd_loss'] = rd_loss
        loss_dict['aux_loss'] = aux_loss

        # calculate psnr
        psnrs = {}
        psnr_left = self.cal_psnr(distortion=dist_left)
        psnr_right =self.cal_psnr(distortion=dist_right)
        psnrs['psnr_left'] = psnr_left
        psnrs['psnr_right'] = psnr_right
        self.avg_psnr = (psnr_left + psnr_right) * 0.5

        return cls_loss, reg_loss, rd_loss, aux_loss, loss_dict, bpps, psnrs, Pictures
        #return cls_loss, reg_loss, loss_dict

    # def forward(self, left_images, right_images, P2, P3, entropy_coding = False): # Actual entropy coding flag
    def test_forward(self, left_images, right_images, P2, P3, entropy_coding =  False): # Actual entropy coding flag
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        if entropy_coding:
            output_dict = self.core.encode_decode(torch.cat([left_images, right_images], dim=1))
        else:
            output_dict = self.core(torch.cat([left_images, right_images], dim=1))

        #output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)

        # 添加bpp loss 和 aux loss
        # calcualte bpp
        num_pixels = left_images.shape[0] * left_images.shape[2] * left_images.shape[3]

        if not entropy_coding:
            bpp_feat_l, bpp_feat_r, bpp_hyper_l, bpp_hyper_r = 0, 0, 0, 0
            for likelihood in output_dict['result']['feat_likelihoods'][0]:
                bpp_feat_l = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_feat_l
            for likelihood in output_dict['result']['feat_likelihoods'][1]:
                bpp_feat_r = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_feat_r
            for likelihood in output_dict['result']['hyper_likelihoods'][0]:
                bpp_hyper_l = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_hyper_l
            for likelihood in output_dict['result']['hyper_likelihoods'][1]:
                bpp_hyper_r = self.cal_bpp(likelihoods=likelihood, num_pixels=num_pixels) + bpp_hyper_r
        else:
            bpp_hyper_l =  0
            bpp_feat_l = 0
            bpp_feat_r = 0
            bpp_hyper_r = 0
        
        self.bpp_total = (bpp_feat_l + bpp_feat_r + bpp_hyper_l + bpp_hyper_r) / 2 

        # calculate distortion # 注意归一化计算的问题
        Pictures = {}
        dist_left, rec_left, ori_left = self.dis(output_dict['left_images_recon'], left_images)
        dist_right, rec_right, rec_right = self.dis(output_dict['right_images_recon'], right_images)
        Pictures['rec_left'] = rec_left
        Pictures['ori_left'] = ori_left
        Pictures['rec_right'] = rec_right
        Pictures['rec_right'] = rec_right
        self.dist_total = dist_left + dist_right

        # calculate psnr
        psnr_left = self.cal_psnr(distortion=dist_left)
        psnr_right =self.cal_psnr(distortion=dist_right)
        self.avg_psnr = (psnr_left + psnr_right) * 0.5

        return scores, bboxes, cls_indexes, self.bpp_total, self.avg_psnr


    # def load_state_dict(self, state_dict, strict):
    #     update_registered_buffers(
    #         self.core.stereo_compression.gaussian_l,
    #         "core.stereo_compression.gaussian_l",
    #         ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
    #         state_dict,
    #     )
    #     update_registered_buffers(
    #         self.core.stereo_compression.gaussian_r,
    #         "core.stereo_compression.gaussian_r",
    #         ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
    #         state_dict,
    #     )
    #     update_registered_buffers(
    #         self.core.stereo_compression.entropy_bottleneck_r,
    #         "core.stereo_compression.entropy_bottleneck_r",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict, strict=strict)


    # def load_state_dict(self, state_dict, strict):
    #     update_registered_buffers(
    #         self.core.stereo_compression.codec_l.codec_1.gaussian,
    #         "core.stereo_compression.codec_l.codec_1.gaussian",
    #         ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
    #         state_dict,
    #     )
    #     update_registered_buffers(
    #         self.core.stereo_compression.gaussian_r,
    #         "core.stereo_compression.gaussian_r",
    #         ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
    #         state_dict,
    #     )
    #     update_registered_buffers(
    #         self.core.stereo_compression.entropy_bottleneck_r,
    #         "core.stereo_compression.entropy_bottleneck_r",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict, strict=strict)

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)
    
    @ staticmethod
    def cal_bpp(likelihoods: torch.Tensor, num_pixels: int):
        bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        return bpp
    
    @ staticmethod
    def cal_rd_cost(distortion: torch.Tensor, bpp: torch.Tensor, lambda_weight: float):
        rd_cost = lambda_weight * distortion + bpp
        return rd_cost
    
    @ staticmethod
    def cal_psnr(distortion: torch.Tensor):
        psnr = -10 * torch.log10(distortion)
        return psnr

# added jdc
@DETECTOR_DICT.register_module
class Stereo3D_VCM_Ablation(Stereo3D_VCM):
    """
        Stereo3D_VCM_Image
    """
    def build_core(self, network_cfg):
        # self.core = YoloStereo3DCore_VCM_Ablation(network_cfg.backbone)
        self.core = YoloStereo3DCore_VCM_Ablation_2(network_cfg.backbone)

# added jdc
@DETECTOR_DICT.register_module
class Stereo3D_VCM_Ablation_sfatten(Stereo3D_VCM):
    """
        Stereo3D_VCM_Image
    """
    def build_core(self, network_cfg):
        # self.core = YoloStereo3DCore_VCM_Ablation(network_cfg.backbone)
        self.core = YoloStereo3DCore_VCM_Ablation_sfatten(network_cfg.backbone)

# added jdc
@DETECTOR_DICT.register_module
class Stereo3D_VCM_MFCNet(Stereo3D_VCM):
    """
        Stereo3D_VCM_Image
    """
    def build_core(self, network_cfg):
        # self.core = YoloStereo3DCore_VCM_Ablation(network_cfg.backbone)
        self.core = YoloStereo3DCore_VCM_MFCNet(network_cfg.backbone)

if __name__ == "__main__":
    from thop import profile
    import sys
    from easydict import EasyDict as edict
    from visualDet3D.utils.utils import cfg_from_file
    config = '/media/D/visualDet3D-master/config/Stereo3D_VCM_MFCNet.py'
    cfg = cfg_from_file(config)
    x = torch.randn(1, 3, 288, 1280).cuda()
    y = torch.randn(1, 3, 288, 1280).cuda()
    p2 = torch.randn(1, 3, 4).cuda()
    p3 = torch.randn(1, 3, 4).cuda()

    a = Stereo3D_VCM_MFCNet(cfg.detector, cfg.data.train_augmentation).cuda()
    #a = Stereo3D_VCM_Image(cfg.detector, cfg.data.train_augmentation).cuda()
    input = (x, y, p2, p3, 0)
    flops, params = profile(a, (input), verbose=False)
    print(" Codec|FLOPs: %sG |Params: %sM" % (flops / 1e9, params / 1e6))