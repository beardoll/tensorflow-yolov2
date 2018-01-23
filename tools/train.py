import tensorflow as tf
import _init_paths
from net.yolov2_net import YOLOv2_net
import os
from dataset.factory import get_imdb
from config.config import cfg
from solver.yolov2_solver import SolverWrapper
import numpy as np

DATASET = 'kitti'
#DATASET = 'pascal_voc'

def train():
    pretrained_model = os.path.join(cfg.PRETRAINED_DIR, 'npy', 'yolov2.npy')
    assert os.path.exists(pretrained_model), \
            'Model path {} does not exist!'.format(pretrained_model)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    name = DATASET + '_train'
    snapshot_infix = DATASET

    output_dir = os.path.join(cfg.TRAIN.TRAINED_DIR, DATASET)

    imdb = get_imdb(name)

    sw = SolverWrapper(imdb, snapshot_infix, output_dir, pretrained_model)
    
    sw.train_net()


if __name__ == '__main__':
    train()
