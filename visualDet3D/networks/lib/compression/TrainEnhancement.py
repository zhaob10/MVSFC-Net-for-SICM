import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from pytorch_msssim import ms_ssim

from Modules.Dataset import H5Dataset
from Modules.Utils import init, cal_rd_cost, cal_psnr, cal_bpp
from Codec import DSIC
from Enhance import EnhanceNet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self):
        self.args, self.logger, self.checkpoints_dir, self.tensorboard = init()

        self.codec = DSIC().to("cuda" if self.args.gpu else "cpu")
        self.enhance = EnhanceNet().to("cuda" if self.args.gpu else "cpu")
        self.dist_metric = nn.MSELoss().to("cuda" if self.args.gpu else "cpu")
        self.optimizer = Adam([{'params': self.enhance.parameters(), 'initial_lr': self.args.lr}], lr=self.args.lr)

        self.train_dataset = H5Dataset(h5_file=self.args.train_dataset)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch, shuffle=True, pin_memory=True)
        self.eval_dataset = H5Dataset(h5_file=self.args.eval_dataset)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=1, shuffle=False, pin_memory=True)
        self.train_steps = self.eval_steps = 0

    def train(self):
        start_epoch = self.load_checkpoints()
        self.codec.eval()
        if self.args.fine_tune:
            scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[100, 200, 300, 400],
                                    gamma=0.5, last_epoch=start_epoch - 1)
        else:
            scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[200, 400, 600, 800],
                                    gamma=0.5, last_epoch=start_epoch - 1)

        for epoch in range(start_epoch, self.args.max_epoch):
            print("\nEpoch {0}".format(str(epoch)))
            self.train_one_epoch()
            scheduler.step()
            if epoch % self.args.eval_epochs == 0:
                self.eval()
            if epoch % self.args.save_epochs == 0:
                self.save_ckpt(epoch=epoch)

    @torch.no_grad()
    def eval(self):
        self.enhance.eval()
        local_step = 0
        avg_ms_ssim_l = avg_ms_ssim_r = avg_psnr_r = avg_psnr_l = avg_delta_psnr_l = avg_delta_psnr_r = 0.0
        for _, frames_tensor in tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), smoothing=0.9):
            img_l = frames_tensor[:, 1, :, :, :].to("cuda" if self.args.gpu else "cpu")
            img_r = frames_tensor[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")

            # compress
            img_l_hat, img_r_hat, _, _, _, _ = self.codec(img_l, img_r)
            psnr_left = cal_psnr(self.dist_metric(img_l_hat, img_l))
            psnr_right = cal_psnr(self.dist_metric(img_r_hat, img_r))

            # enhancement
            img_l_hat, img_r_hat = self.enhance(img_l_hat, img_r_hat)        
            psnr_left_enhance = cal_psnr(self.dist_metric(img_l_hat, img_l))
            psnr_right_enhance = cal_psnr(self.dist_metric(img_r_hat, img_r))
            
            # calculate ms-ssim
            ms_ssim_left = ms_ssim(X=img_l_hat, Y=img_l, data_range=1, size_average=True)
            ms_ssim_right = ms_ssim(X=img_r_hat, Y=img_r, data_range=1, size_average=True)

            # record
            local_step += 1
            avg_ms_ssim_l += ms_ssim_left
            avg_ms_ssim_r += ms_ssim_right
            avg_psnr_r += psnr_right_enhance
            avg_psnr_l += psnr_left_enhance
            avg_delta_psnr_l += psnr_left_enhance - psnr_left
            avg_delta_psnr_r += psnr_right_enhance - psnr_right

        print("\nMS_SSIM_L={:.4f}".format(avg_ms_ssim_l / local_step) +
              " MS_SSIM_R={:.4f}".format(avg_ms_ssim_r / local_step) +
              " MS_SSIM_Average={:.4f}".format((avg_ms_ssim_l + avg_ms_ssim_r) / local_step / 2) +
              " PSNR_L={:.2f}".format(avg_psnr_l / local_step) +
              " PSNR_R={:.2f}".format(avg_psnr_r / local_step) +
              " PSNR_Average={:.2f}".format((avg_psnr_r + avg_psnr_l) / local_step / 2) +
              " Delta_PSNR_L={:.2f}".format(avg_delta_psnr_l / local_step) +
              " Delta_PSNR_R={:.2f}".format(avg_delta_psnr_r / local_step) 
             )

    def train_one_epoch(self):
        self.enhance.train()
        for _, frames_tensor in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), smoothing=0.9):
            img_l = frames_tensor[:, 1, :, :, :].to("cuda" if self.args.gpu else "cpu")
            img_r = frames_tensor[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")
            with torch.no_grad():
                img_l_hat, img_r_hat, _, _, _, _ = self.codec(img_l, img_r)
            img_l_hat, img_r_hat = self.enhance(img_l_hat, img_r_hat)    

            dist_left = self.dist_metric(img_l_hat, img_l)
            dist_right = self.dist_metric(img_r_hat, img_r)
            dist_total = dist_left + dist_right
            
            dist_total.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_ckpt(self, epoch: int):
        checkpoint = {
            "codec": self.codec.state_dict(),
            "enhance": self.enhance.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, '%s/enhance_%.3d_%.3d.pth' % (self.checkpoints_dir, self.args.lambda_weight, epoch))
        print("\n======================Saving model {0}======================".format(str(epoch)))


    def load_checkpoints(self):
        if self.args.checkpoints:
            print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
            ckpt = torch.load(self.args.checkpoints)
            # load  weights
            self.codec.load_state_dict(ckpt["codec"])
            self.enhance.load_state_dict(ckpt["enhance"])
            # load optimizer params
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except:
                print("Can not find some optimizers params, just ignore")
            start_epoch = ckpt["epoch"] + 1
        elif self.args.pretrained:
            ckpt = torch.load(self.args.pretrained)
            print("\n===========Load codecwork weights {0}===========\n".format(self.args.pretrained))
            self.codec.load_state_dict(ckpt["codec"])
            try:
                self.enhance.load_state_dict(ckpt["enhance"])
            except:
                print("Cannot find weights for enhancement net")
            start_epoch = 0
        else:
            print("biu biu biu")
            exit(1)
        return start_epoch


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
