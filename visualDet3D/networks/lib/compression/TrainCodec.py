import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from torch.nn import DataParallel
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Codec import Codec
from Modules.Dataset import H5Dataset
from Modules.Utils import init, cal_psnr, cal_bpp, cal_rd_cost

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self):
        self.args, self.logger, self.checkpoints_dir = init()
        self.device_idx = range(torch.cuda.device_count())

        self.net = Codec().to("cuda" if self.args.gpu else "cpu")

        self.dist_metric = nn.MSELoss().to("cuda" if self.args.gpu else "cpu")
        self.optimizer, self.aux_optimizer = self.init_optimizer()

        self.train_dataset = H5Dataset(h5_file=self.args.train_dataset)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch, shuffle=True,
                                           pin_memory=True)
        self.eval_dataset = H5Dataset(h5_file=self.args.eval_dataset)
        self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=1, shuffle=False, pin_memory=True)

        self.train_steps = self.eval_steps = 0

    def train(self):
        start_epoch = self.load_checkpoints()

        scheduler, aux_scheduler = self.init_scheduler(start_epoch=start_epoch)
        
        if len(self.device_idx) > 1:
            self.net = DataParallel(self.net)
            print("==================Using GPUs " + str(list(self.device_idx)) + " =======================")
        
        for epoch in range(start_epoch, self.args.max_epoch):
            msg = "\nEpoch {0}".format(str(epoch))
            self.logger.info(msg)
            self.train_one_epoch()
            scheduler.step()
            aux_scheduler.step()
            if epoch % self.args.eval_epochs == 0:
                self.eval()
            if epoch % self.args.save_epochs == 0:
                self.save_ckpt(epoch=epoch)

    @torch.no_grad()
    def eval(self):
        self.net.eval()
        local_step = 0
        avg_ms_ssim_l = avg_ms_ssim_r = avg_psnr_r = avg_psnr_l = avg_bpp_feats_l = avg_bpp_feats_r = \
            avg_bpp_hyper_l = avg_bpp_hyper_r = 0.0
        for _, frames_tensor in tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), smoothing=0.9, ncols=50):
            img_l = frames_tensor[:, 1, :, :, :].to("cuda" if self.args.gpu else "cpu")
            img_r = frames_tensor[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")

            output = self.net(img_l, img_r)

            # calculate distortion
            dist_left = self.dist_metric(output['img_hat'][0], img_l)
            dist_right = self.dist_metric(output['img_hat'][1], img_r)
            # calculate bpp
            num_pixels = img_l.shape[0] * img_l.shape[2] * img_l.shape[3]
            bpp_feats_l = cal_bpp(likelihoods=output['feat_likelihoods'][0], num_pixels=num_pixels)
            bpp_feats_r = cal_bpp(likelihoods=output['feat_likelihoods'][1], num_pixels=num_pixels)
            bpp_hyper_l = cal_bpp(likelihoods=output['hyper_likelihoods'][0], num_pixels=num_pixels)
            bpp_hyper_r = cal_bpp(likelihoods=output['hyper_likelihoods'][1], num_pixels=num_pixels)

            # visualization
            psnr_left = cal_psnr(dist_left)
            psnr_right = cal_psnr(dist_right)

            # calculate ms-ssim
            ms_ssim_left = ms_ssim(X=output['img_hat'][0], Y=img_l, data_range=1, size_average=True)
            ms_ssim_right = ms_ssim(X=output['img_hat'][1], Y=img_r, data_range=1, size_average=True)

            # record
            local_step += 1
            avg_ms_ssim_l += ms_ssim_left
            avg_ms_ssim_r += ms_ssim_right
            avg_psnr_r += psnr_right
            avg_psnr_l += psnr_left
            avg_bpp_feats_l += bpp_feats_l
            avg_bpp_feats_r += bpp_feats_r
            avg_bpp_hyper_l += bpp_hyper_l
            avg_bpp_hyper_r += bpp_hyper_r

        msg = "\nMS_SSIM_L={:.4f}".format(avg_ms_ssim_l / local_step) + \
              " MS_SSIM_R={:.4f}".format(avg_ms_ssim_r / local_step) + \
              " MS_SSIM_Average={:.4f}".format((avg_ms_ssim_l + avg_ms_ssim_r) / local_step / 2) + \
              " PSNR_L={:.4f}".format(avg_psnr_l / local_step) + \
              " PSNR_R={:.4f}".format(avg_psnr_r / local_step) + \
              " PSNR_Average={:.4f}".format((avg_psnr_r + avg_psnr_l) / local_step / 2) + \
              " Bpp_L={:.4f}".format(avg_bpp_feats_l / local_step) + \
              " Bpp_R={:.4f}".format(avg_bpp_feats_r / local_step) + \
              " Hyper_Bpp_L={:.4f}".format(avg_bpp_hyper_l / local_step) + \
              " Hyper_Bpp_R={:.4f}".format(avg_bpp_hyper_r / local_step) + \
              " Bpp_Total_L={:.4f}".format((avg_bpp_feats_l + avg_bpp_hyper_l) / local_step) + \
              " Bpp_Total_R={:.4f}".format((avg_bpp_feats_r + avg_bpp_hyper_r) / local_step) + \
              " Bpp_Average={:.4f}".format(
                  (avg_bpp_feats_l + avg_bpp_hyper_l + avg_bpp_feats_r + avg_bpp_hyper_r) / local_step / 2)
        self.logger.info(msg)

    def train_one_epoch(self):
        self.net.train()
        for idx, frames_tensor in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), ncols=50, smoothing=0.9):
            img_l = frames_tensor[:, 1, :, :, :].to("cuda" if self.args.gpu else "cpu")
            img_r = frames_tensor[:, 0, :, :, :].to("cuda" if self.args.gpu else "cpu")

            output = self.net(img_l, img_r)

            # calculate distortion
            dist_left = self.dist_metric(output['img_hat'][0], img_l)
            dist_right = self.dist_metric(output['img_hat'][1], img_r)
            dist_total = dist_left + dist_right

            # calculate bpp
            num_pixels = img_l.shape[0] * img_l.shape[2] * img_l.shape[3]
            bpp_feats_l = cal_bpp(likelihoods=output['feat_likelihoods'][0], num_pixels=num_pixels)
            bpp_feats_r = cal_bpp(likelihoods=output['feat_likelihoods'][1], num_pixels=num_pixels)
            bpp_hyper_l = cal_bpp(likelihoods=output['hyper_likelihoods'][0], num_pixels=num_pixels)
            bpp_hyper_r = cal_bpp(likelihoods=output['hyper_likelihoods'][1], num_pixels=num_pixels)
            bpp_total = bpp_feats_l + bpp_feats_r + bpp_hyper_l + bpp_hyper_r
            
            # calculate rd loss and auxiliary loss
            rd_cost = cal_rd_cost(distortion=dist_total, bpp=bpp_total, lambda_weight=self.args.lambda_weight)
            aux_loss = self.net.module.aux_loss()
            
            # backward
            rd_cost.backward()
            aux_loss.backward()

            if self.args.batch >= self.args.expect_batch or (idx + 1) * self.args.batch % self.args.expect_batch == 0:
                self.optimizer.step()
                self.aux_optimizer.step()
                self.optimizer.zero_grad()
                self.aux_optimizer.zero_grad()

    def save_ckpt(self, epoch: int):
        checkpoint = {
            "codec": self.net.module.state_dict() if len(self.device_idx) > 1 else self.net.state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict()
        }

        torch.save(checkpoint, '%s/model_%.3d_%.3d.pth' % (self.checkpoints_dir, self.args.lambda_weight, epoch))
        msg = "\n======================Saving model {0}======================".format(str(epoch))
        self.logger.info(msg)

    def init_optimizer(self):
        assert self.net is not None, "Network must be instantiated first"

        parameters = set(n for n, p in self.net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
        aux_parameters = set(n for n, p in self.net.named_parameters() if n.endswith(".quantiles") and p.requires_grad)

        # Make sure there is no intersection of parameters
        params_dict = dict(self.net.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters
        assert len(inter_params) == 0, "intersection between main params and auxiliary params"
        assert len(union_params) - len(params_dict.keys()) == 0, "intersection between main params and auxiliary params"
        optimizer = Adam([{'params': (params_dict[n] for n in sorted(list(parameters))), 'initial_lr': self.args.lr}],
                         lr=self.args.lr)
        aux_optimizer = Adam(
            [{'params': (params_dict[n] for n in sorted(list(aux_parameters))), 'initial_lr': 10 * self.args.lr}],
            lr=10 * self.args.lr)
        return optimizer, aux_optimizer

    def init_scheduler(self, start_epoch: int):
        lr_decay_times = self.args.lr_decay_times
        lr_decay_base_epochs = self.args.max_epoch // (lr_decay_times + 1)

        scheduler = MultiStepLR(optimizer=self.optimizer,
                                milestones=[lr_decay_base_epochs * i for i in range(1, lr_decay_times + 1)],
                                gamma=0.5, last_epoch=start_epoch - 1)
        aux_scheduler = MultiStepLR(optimizer=self.aux_optimizer,
                                    milestones=[lr_decay_base_epochs * i for i in range(1, lr_decay_times + 1)],
                                    gamma=0.5, last_epoch=start_epoch - 1)
        return scheduler, aux_scheduler

    def load_checkpoints(self):
        if self.args.checkpoints:
            print("\n===========Load checkpoints {0}===========\n".format(self.args.checkpoints))
            ckpt = torch.load(self.args.checkpoints, map_location="cuda" if self.args.gpu else "cpu")
            # load codec weights
            self.net.load_state_dict(ckpt["codec"])
            # load optimizer params
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
            except:
                print("Can not find some optimizers params, just ignore")
            start_epoch = ckpt["epoch"] + 1
        elif self.args.pretrained:
            ckpt = torch.load(self.args.pretrained)
            print("\n===========Load network weights {0}===========\n".format(self.args.pretrained))
            # load codec weights
            pretrained_dict = ckpt["codec"]
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)
            start_epoch = 0
        else:
            print("\n===========Training from scratch===========\n")
            start_epoch = 0
        return start_epoch


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
