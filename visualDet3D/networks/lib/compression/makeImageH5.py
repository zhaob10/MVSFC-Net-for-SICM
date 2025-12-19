import os
import h5py
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

from pathlib import Path


def split(ori_size: tuple, crop_size: tuple):
    clip_boxes = []
    ori_height, ori_width = ori_size
    crop_height, crop_width = crop_size

    num_boxes_hor = ori_width // crop_width
    num_boxes_ver = ori_height // crop_height

    assert num_boxes_ver > 0 and num_boxes_hor > 0

    for i in range(num_boxes_hor):
        for j in range(num_boxes_ver):
            clip_box = (i * crop_width, j * crop_height, (i + 1) * crop_width, (j + 1) * crop_height)
            clip_boxes.append(clip_box)
    return clip_boxes


def make_kitti_h5(kitti_folder: str, h5_folder: str, crop_size: tuple = (512, 512)):
    experiment_dir = Path(h5_folder)
    experiment_dir.mkdir(exist_ok=True)
    # create training set
    print("\n====================Creating training set========================\n")
    with h5py.File(os.path.join(h5_folder, "Train_KITTI_{0}.hdf5".format(str(crop_size[0]))), mode='w') as f:
        # traverse all images
        img_list = []
        left_view_folder = os.path.join(kitti_folder, "train", "left")
        right_view_folder = os.path.join(kitti_folder, "train", "right")
        for img_filename in os.listdir(left_view_folder):
            img_path_l = os.path.join(left_view_folder, img_filename)
            img_path_r = os.path.join(right_view_folder, img_filename)
            if os.path.isfile(img_path_l) and os.path.isfile(img_path_r):  # avoid some single image
                img_list.append([img_path_l, img_path_r])
        img_list.sort()
        # split image into patches
        img_counter = 0
        with tqdm(img_list) as bar:
            for img_path_l, img_path_r in bar:
                bar.set_postfix(Processing=os.path.split(img_path_l)[-1])
                img_l = Image.open(img_path_l)
                img_r = Image.open(img_path_r)
                crop_boxes = split(ori_size=(img_r.size[1], img_r.size[0]), crop_size=crop_size)
                for crop_box in crop_boxes:
                    img_counter += 1
                    patch_l = np.array(img_l.crop(crop_box))
                    patch_r = np.array(img_r.crop(crop_box))
                    f.create_dataset(name=str(img_counter), data=np.array([patch_r, patch_l]),
                                     compression="gzip", compression_opts=4)
                img_r.close()
                img_l.close()

    # create testing set
    print("\n====================Creating testing set========================\n")
    with h5py.File(os.path.join(h5_folder, "Eval_KITTI.hdf5"), mode='w') as f:
        img_list = []
        left_view_folder = os.path.join(kitti_folder, "test", "left")
        right_view_folder = os.path.join(kitti_folder, "test", "right")
        for img_filename in os.listdir(left_view_folder):
            img_path_l = os.path.join(left_view_folder, img_filename)
            img_path_r = os.path.join(right_view_folder, img_filename)
            if os.path.isfile(img_path_l) and os.path.isfile(img_path_r):  # avoid some single image
                img_list.append([img_path_l, img_path_r])
        img_list.sort()
        img_counter = 0
        with tqdm(img_list) as bar:
            for img_path_l, img_path_r in bar:
                bar.set_postfix(Processing=os.path.split(img_path_l)[-1])
                img_l = Image.open(img_path_l)
                img_r = Image.open(img_path_r)
                f.create_dataset(name=str(img_counter), data=np.array([np.array(img_r), np.array(img_l)]),
                                 compression="gzip", compression_opts=4)
                img_r.close()
                img_l.close()
                img_counter += 1


if __name__ == "__main__":
    kitti_folder = "/media/xrliu/data/Dataset/HESIC/InDoor"
    make_kitti_h5(kitti_folder, h5_folder='./H5Img')
