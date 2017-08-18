import tensorflow as tf
import os.path as osp
from config.config import cfg

#pwd = os.getcwd()
#filename = osp.join(pwd, 'region.so')

filename = osp.join(cfg.ROOT_DIR, 'lib', 'region_layer', 'region.so')

assert osp.exists(filename), \
        'Path {} does not exist!'.format(filename)

_region_module = tf.load_op_library(filename)
region = _region_module.region
region_grad = _region_module.region_grad
