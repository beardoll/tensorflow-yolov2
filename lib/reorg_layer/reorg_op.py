import tensorflow as tf
import os
import numpy as np

filename = os.path.join(os.getcwd(), 'reorg.so')
assert os.path.exists(filename)

_reorg_module = tf.load_op_library(filename)
reorg = _reorg_module.reorg
reorg_grad = _reorg_module.reorg_grad


