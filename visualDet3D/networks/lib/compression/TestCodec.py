import time
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from tqdm import tqdm

from Codec import Codec
from Modules.Utils import cal_psnr, cal_bpp, write_to_file, read_from_file

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


class Tester:
    def __init__(self):
        self.args = self.parse_args()
        self.codec = Codec().to("cuda" if self.args.gpu else "cpu").eval()
        self.dist_metric = nn.MSELoss().to("cuda" if self.args.gpu else "cpu")

    def test(self):
        self.load_models()
        self.codec.update()
        print("\n===========Test images from folder {0}===========".format(self.args.img_folder))
        if self.args.use_codec:
            self.test_imgs_real()
        else:
            self.test_imgs()

    @torch.no_grad()
    def test_imgs_real(self):
        assert self.args.save_bin is not None, "Must give the binary file path"
        save_bin = Path(self.args.save_bin)
        save_bin.mkdir(exist_ok=True)
        save_rec = Path(self.args.save_recon)
        save_rec.mkdir(exist_ok=True)

        imgs = self.load_imgs()
        avg_ms_ssim_l = avg_ms_ssim_r = avg_psnr_r = avg_psnr_l = avg_bpp = avg_enc_time = avg_dec_time = step = 0.0

        for idx, imgs_pair in tqdm(enumerate(imgs), total=len(imgs), smoothing=0.9):
            img_l = imgs_pair[0].to("cuda" if self.args.gpu else "cpu")
            img_r = imgs_pair[1].to("cuda" if self.args.gpu else "cpu")

            torch.cuda.synchronize()
            start = time.time()
            enc_results = self.codec.compress(img_left=img_l, img_right=img_r)
            torch.cuda.synchronize()
            end = time.time()
            avg_enc_time += end - start

            bin_path = os.path.join(self.args.save_bin, "{0}.bin".format(str(idx)))
            bpp = write_to_file(file_path=bin_path, img_size=img_l.shape[-2:], hyper_shape=enc_results["shape"],
                                strings_l=enc_results["strings"][0], strings_r=enc_results["strings"][1],
                                hyper_strings_l=enc_results["strings"][2], hyper_strings_r=enc_results["strings"][3])
            # NOTE: next u can use decoded results from bin file (actually no difference)
            torch.cuda.synchronize()
            start = time.time()
            dec_results = self.codec.decompress(strings=enc_results["strings"], shape=enc_results["shape"])
            torch.cuda.synchronize()
            end = time.time()
            avg_dec_time += end - start
            img_hat_l, img_hat_r = dec_results["img_hat"]

            # calculate PSNR
            dist_left = self.dist_metric(img_hat_l, img_l)
            dist_right = self.dist_metric(img_hat_r, img_r)
            psnr_left = cal_psnr(dist_left)
            psnr_right = cal_psnr(dist_right)

            # calculate MS-SSIM
            ms_ssim_left = ms_ssim(X=img_hat_l, Y=img_l, data_range=1, size_average=True)
            ms_ssim_right = ms_ssim(X=img_hat_r, Y=img_r, data_range=1, size_average=True)

            # gather info
            step += 1
            avg_bpp += bpp
            avg_ms_ssim_l += ms_ssim_left
            avg_ms_ssim_r += ms_ssim_right
            avg_psnr_r += psnr_right
            avg_psnr_l += psnr_left

            # save reconstructed images
            if self.args.save_recon is not None:
                trans = transforms.ToPILImage()
                img_hat_l = img_hat_l.cpu().clone()[0]
                img_hat_l = trans(img_hat_l)
                img_hat_l.save(os.path.join(self.args.save_recon, "rec_{0}_left.png".format(str(idx))))
                img_hat_r = img_hat_r.cpu().clone()[0]
                img_hat_r = trans(img_hat_r)
                img_hat_r.save(os.path.join(self.args.save_recon, "rec_{0}_right.png".format(str(idx))))

        print("\nMS_SSIM_L={:.4f}".format(avg_ms_ssim_l / step) +
              " MS_SSIM_R={:.4f}".format(avg_ms_ssim_r / step) +
              " MS_SSIM_Average={:.4f}".format((avg_ms_ssim_l + avg_ms_ssim_r) / step / 2) +
              " PSNR_L={:.4f}".format(avg_psnr_l / step) +
              " PSNR_R={:.4f}".format(avg_psnr_r / step) +
              " PSNR_Average={:.4f}".format((avg_psnr_r + avg_psnr_l) / step / 2) +
              " Bpp={:.4f}".format(avg_bpp / step) +
              " Encode Time={:.4f}".format(avg_enc_time / step) +
              " Decode Time={:.4f}".format(avg_dec_time / step)
              )

    @torch.no_grad()
    def test_imgs(self):
        save_rec = Path(self.args.save_recon)
        save_rec.mkdir(exist_ok=True)

        imgs = self.load_imgs()
        avg_ms_ssim_l = avg_ms_ssim_r = avg_psnr_r = avg_psnr_l \
            = avg_bpp_main_l = avg_bpp_main_r = avg_bpp_hyper_l = avg_bpp_hyper_r = step = avg_time = 0.0

        for idx, imgs_pair in tqdm(enumerate(imgs), total=len(imgs), smoothing=0.9):
            img_l = imgs_pair[0].to("cuda" if self.args.gpu else "cpu")
            img_r = imgs_pair[1].to("cuda" if self.args.gpu else "cpu")

            # encode images
            torch.cuda.synchronize()
            start = time.time()
            output = self.codec(img_l, img_r)
            end = time.time()
            avg_time += end - start
            # calculate PSNR
            dist_left = self.dist_metric(output['img_hat'][0], img_l)
            dist_right = self.dist_metric(output['img_hat'][1], img_r)
            psnr_left = cal_psnr(dist_left)
            psnr_right = cal_psnr(dist_right)

            # calculate bpp
            num_pixels = img_l.shape[0] * img_l.shape[2] * img_l.shape[3]
            bpp_feats_l = cal_bpp(likelihoods=output['feat_likelihoods'][0], num_pixels=num_pixels)
            bpp_feats_r = cal_bpp(likelihoods=output['feat_likelihoods'][1], num_pixels=num_pixels)
            bpp_hyper_l = cal_bpp(likelihoods=output['hyper_likelihoods'][0], num_pixels=num_pixels)
            bpp_hyper_r = cal_bpp(likelihoods=output['hyper_likelihoods'][1], num_pixels=num_pixels)

            # calculate MS-SSIM
            ms_ssim_left = ms_ssim(X=output['img_hat'][0], Y=img_l, data_range=1, size_average=True)
            ms_ssim_right = ms_ssim(X=output['img_hat'][1], Y=img_r, data_range=1, size_average=True)

            # gather info
            step += 1
            avg_ms_ssim_l += ms_ssim_left
            avg_ms_ssim_r += ms_ssim_right
            avg_psnr_r += psnr_right
            avg_psnr_l += psnr_left
            avg_bpp_main_l += bpp_feats_l
            avg_bpp_main_r += bpp_feats_r
            avg_bpp_hyper_l += bpp_hyper_l
            avg_bpp_hyper_r += bpp_hyper_r

            # save reconstructed images
            if self.args.save_recon is not None:
                trans = transforms.ToPILImage()
                img_hat_l = output['img_hat'][0].cpu().clone()[0]  # shape (C, H, W)
                img_hat_l = trans(img_hat_l)
                img_hat_l.save(os.path.join(self.args.save_recon, "rec_{0}_left.png".format(str(idx))))
                img_hat_r = output['img_hat'][1].cpu().clone()[0]  # shape (C, H, W)
                img_hat_r = trans(img_hat_r)
                img_hat_r.save(os.path.join(self.args.save_recon, "rec_{0}_right.png".format(str(idx))))

        print("\nMS_SSIM_L={:.4f}".format(avg_ms_ssim_l / step) +
              " MS_SSIM_R={:.4f}".format(avg_ms_ssim_r / step) +
              " MS_SSIM_Average={:.4f}".format((avg_ms_ssim_l + avg_ms_ssim_r) / step / 2) +
              " PSNR_L={:.4f}".format(avg_psnr_l / step) +
              " PSNR_R={:.4f}".format(avg_psnr_r / step) +
              " PSNR_Average={:.4f}".format((avg_psnr_r + avg_psnr_l) / step / 2) +
              " Bpp_L={:.4f}".format(avg_bpp_main_l / step) +
              " Bpp_R={:.4f}".format(avg_bpp_main_r / step) +
              " Hyper_Bpp_L={:.4f}".format(avg_bpp_hyper_l / step) +
              " Hyper_Bpp_R={:.4f}".format(avg_bpp_hyper_r / step) +
              " Bpp_Total_L={:.4f}".format((avg_bpp_main_l + avg_bpp_hyper_l) / step) +
              " Bpp_Total_R={:.4f}".format((avg_bpp_main_r + avg_bpp_hyper_r) / step) +
              " Bpp_Average={:.4f}".format(
                  (avg_bpp_main_l + avg_bpp_hyper_l + avg_bpp_main_r + avg_bpp_hyper_r) / step / 2) +
              " Encoding and decoding time={:.4f}".format(avg_time / step))

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--img_folder", type=str, default=None, help="Folder of images to be encoded")
        parser.add_argument("--save_recon", type=str, default=None, help="Folder to save reconstructed images folder")
        parser.add_argument("--save_bin", type=str, default=None, help="Folder to save bitstream")

        parser.add_argument("--gpu", action='store_true', default=False, help="use gpu or cpu")
        parser.add_argument("--model", type=str, default="./Models/Codec_4096.pth", help="Model file path")
        parser.add_argument("--use_codec", action='store_true', default=False, help="use gpu or cpu")

        args = parser.parse_args()
        return args

    def load_imgs(self):
        trans = transforms.Compose([transforms.ToTensor(), ])
        imgs = []
        left_folder = os.path.join(self.args.img_folder, "left")
        right_folder = os.path.join(self.args.img_folder, "right")
        for img_name in os.listdir(left_folder):
            img_l = trans(Image.open(os.path.join(left_folder, img_name))).unsqueeze(dim=0)
            img_r = trans(Image.open(os.path.join(right_folder, img_name))).unsqueeze(dim=0)
            imgs.append([img_l, img_r])
        return imgs

    def load_models(self):
        print("\n===========Load model weights {0}===========\n".format(self.args.model))
        ckpt = torch.load(self.args.model)
        self.codec.load_state_dict(ckpt["codec"])


if __name__ == "__main__":
    tester = Tester()
    tester.test()
