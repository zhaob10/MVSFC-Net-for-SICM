import time
from .kitti_common import get_label_annos, get_compression_results
from .eval import get_official_eval_result, get_coco_eval_result
from numba import cuda

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2", # 训练集左视图的label
             result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
             label_split_file="val.txt", #　'./workdirs/Stereo3D/output/validation/data'
             current_classes=[0],
             gpu=0):
    cuda.select_device(gpu)
    dt_annos = get_label_annos(result_path) 
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = get_label_annos(label_path, val_image_ids)
    result_texts = []
    for current_class in current_classes: # 循环每一个类别
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class))
    return result_texts


def evaluate_VCM(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2", # 训练集左视图的label
             result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
             label_split_file="val.txt", #　'./workdirs/Stereo3D/output/validation/data'
             current_classes=[0],
             gpu=0):
    cuda.select_device(gpu)
    dt_annos = get_label_annos(result_path)

    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = get_label_annos(label_path, val_image_ids)
    result_texts = []

    com_annos = get_compression_results(result_path) 
    result_texts.append(com_annos)
    
    for current_class in current_classes: # 循环每一个类别
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class))
    return result_texts
