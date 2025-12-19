import os
import argparse
import datetime
import logging
import math
import struct
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min_: float = SCALES_MIN, max_: float = SCALES_MAX, levels: int = SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min_), math.log(max_), levels))


def parse_args():
    parser = argparse.ArgumentParser()

    # for loading data
    parser.add_argument("--train_dataset", type=str, default='./Dataset/Train/Train_256.hdf5',
                        help="train dataset h5 file")
    parser.add_argument("--eval_dataset", type=str, default='./Dataset/Train/Eval_256.hdf5',
                        help="eval dataset h5 file")

    # for loading model
    parser.add_argument("--checkpoints", type=str, help="checkpoints file path")
    parser.add_argument("--pretrained", type=str, help="pretrained model path")

    # batch size
    parser.add_argument("--batch", type=int, default=2, help="batch size for Fusion stage")

    # learning rate
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Fusion stage")
    parser.add_argument("--lr_decay_times", type=int, default=4, help="learning rate decay times")
    parser.add_argument("--expect_batch", type=int, default=4, help="expected batch, using in grad accumulation")

    # for training
    parser.add_argument("--fine_tune", action='store_true', default=False, help="fine tune")
    parser.add_argument("--gpu", action='store_true', default=False, help="use gpu or cpu")

    # hyper parameters
    parser.add_argument("--lambda_weight", type=float, default=1024, help="weights for bitrate item")

    # epoch
    parser.add_argument("--max_epoch", type=int, default=200, help="max training epochs")

    # for recording
    parser.add_argument("--verbose", action='store_true', default=False, help="use tensorboard and logger")
    parser.add_argument("--save_epochs", type=int, default=5, help="save model after epochs")
    parser.add_argument("--save_dir", type=str, default="./Experiments", help="directory for recording")
    parser.add_argument("--eval_epochs", type=int, default=1, help="save model after epochs")

    args = parser.parse_args()
    return args


class CustomLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(str(log_dir) + '/Log.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logger = logger

    def info(self, msg, print_: bool = True):
        self.logger.info(msg)
        if print_:
            print(msg)


def init():
    # parse arguments
    args = parse_args()

    # create directory for recording
    experiment_dir = Path(args.save_dir)
    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    experiment_dir.mkdir(exist_ok=True)

    ckpt_dir = experiment_dir.joinpath("Checkpoints/")
    ckpt_dir.mkdir(exist_ok=True)
    print(r"===========Save checkpoints to {0}===========".format(str(ckpt_dir)))

    if args.verbose:
        # initialize logger
        log_dir = experiment_dir.joinpath('Log/')
        logger = CustomLogger(log_dir=log_dir)
        logger.info('PARAMETER ...', print_=False)
        logger.info(args, print_=False)
    else:
        print(r"===========Disable logger to accelerate training===========")
        logger = None

    return args, logger, ckpt_dir


def initialize_weights(net: nn.Module):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def get_aux_optimizer(net: nn.Module, lr: int):
    parameters = set(n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad)
    aux_parameters = set(n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad)

    # make sure there are no intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    aux_optimizer = torch.optim.Adam(params=(params_dict[n] for n in sorted(list(aux_parameters))), lr=lr)
    return aux_optimizer


def cal_rd_cost(distortion: torch.Tensor, bpp: torch.Tensor, lambda_weight: float):
    rd_cost = lambda_weight * distortion + bpp
    return rd_cost


def cal_bpp(likelihoods: torch.Tensor, num_pixels: int):
    bpp = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    return bpp


def cal_psnr(distortion: torch.Tensor):
    psnr = -10 * torch.log10(distortion)
    return psnr


def sizeof(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_to_file(file_path: str, img_size: tuple, hyper_shape: tuple,
                  hyper_strings_l: list, hyper_strings_r: list, strings_l: list, strings_r: list):
    with Path(file_path).open("wb") as f:
        write_uints(f, (hyper_shape[0], hyper_shape[1]))

        len_hyper_l = len(hyper_strings_l[0])
        len_hyper_r = len(hyper_strings_r[0])
        len_feats_l = len(strings_l[0])
        len_feats_r = len(strings_r[0])

        write_uints(f, (len_hyper_l, len_hyper_r, len_feats_l, len_feats_r))

        write_bytes(f, hyper_strings_l[0])
        write_bytes(f, hyper_strings_r[0])
        write_bytes(f, strings_l[0])
        write_bytes(f, strings_r[0])

    avg_bpp = float(sizeof(file_path)) * 8 / (img_size[0] * img_size[1]) / 2
    return avg_bpp


def read_from_file(bin_path: str):
    with Path(bin_path).open("rb") as f:
        hyper_shape = read_uints(f, 2)
        len_hyper_l, len_hyper_r, len_feats_l, len_feats_r = read_uints(f, 4)

        hyper_strings_l = read_bytes(f, len_hyper_l)
        hyper_strings_r = read_bytes(f, len_hyper_r)
        feats_strings_l = read_bytes(f, len_feats_l)
        feats_strings_r = read_bytes(f, len_feats_r)

    return hyper_shape, [feats_strings_l, ], [feats_strings_r, ], [hyper_strings_l, ], [hyper_strings_r, ]


counter = 0


def visualize_features(features: torch.Tensor, save_path: str, features_name: str = None, channel_idx: int = -1):
    global counter

    if channel_idx == -1:
        features = torch.sum(features[0], dim=0).cpu().numpy()
    else:
        features = features[0, channel_idx].cpu().numpy()
    for i in range(features.shape[0] // 8):
        features[i * 8, :] = features.max()
        features[:, i * 8] = features.max()

    plt.figure()
    plt.imshow(features)
    plt.axis("off")
    if features_name is None:
        features_name = str(counter)
    plt.title("Tensor_" + features_name)
    if save_path:
        plt.savefig(os.path.join(save_path, "Tensor_" + features_name + ".png"))
    plt.close()
    counter += 1
