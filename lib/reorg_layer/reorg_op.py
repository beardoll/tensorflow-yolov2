import tensorflow as tf
import os.path as osp
import numpy as np
from config.config import cfg

filename = osp.join(cfg.ROOT_DIR, 'lib', 'reorg_layer', 'reorg.so')
assert osp.exists(filename), \
        'Path {} does not exist!'.format(filename)

_reorg_module = tf.load_op_library(filename)
reorg = _reorg_module.reorg
reorg_grad = _reorg_module.reorg_grad


