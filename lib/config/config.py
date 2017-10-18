'''YOLOv2 config system

This file specifies default options for YOLOv2 (yolo-voc.cfg).
'''

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Get config by from config import cfg
cfg = __C

'''Train options

'''
__C.TRAIN = edict()

# Batch size
__C.TRAIN.BATCH = 32 
__C.TRAIN.SUBDIVISION = 4 

# Learning rate
__C.TRAIN.LEARNING_RATE = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.STEP_SIZE = [40000, 60000]
__C.SCALES = [0.1, 0.1]

# Max objects in a box
__C.TRAIN.MAX_OBJ = 30   # Cannot change!!!

# Iteration
__C.TRAIN.MAX_ITERS = 80200
__C.TRAIN.ITER_THRESH = 80000

# Weight decay
__C.TRAIN.DECAY = 0.0005

# Data argumentation params
__C.TRAIN.SATURATION = 1.5
__C.TRAIN.EXPOSURE = 1.5
__C.TRAIN.HUE = 0.1
__C.TRAIN.JITTER = 0.3

# The time interval between two snapshots
__C.TRAIN.SNAPSHOTS = 5000

# Anchor boxes per pixel in feature map
__C.TRAIN.BOX_NUM = 5

# Coefficency of computation for loss
__C.TRAIN.OBJECT_SCALE = 5
__C.TRAIN.NOOBJECT_SCALE = 1
__C.TRAIN.CLASS_SCALE = 1
__C.TRAIN.COORD_SCALE = 1

# Thresh for selecting delegate boxes
__C.TRAIN.THRESH = 0.6

# Aprior anchors' size
__C.TRAIN.ANCHORS = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, \
                     8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

# Classes
__C.TRAIN.CLASSES = ['Car', 'Pedestrian', 'Cyclist']
__C.TRAIN.DONT_CARE = ['DontCare', 'Misc', 'Person_sitting', 'Truck', 'Tram']

# Boxes per pixel of output feature map
__C.TRAIN.BOX_NUM = 5

# Snapshot filename
__C.TRAIN.SNAPSHOT_INFIX = 'kitti'
__C.TRAIN.SNAPSHOT_PREFIX = 'yolov2'


# Paths
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
__C.PRETRAINED_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'pretrained_model'))
__C.TRAIN.SUMMARY_DIR = osp.abspath(osp.join(__C.OUTPUT_DIR, 'summary'))
__C.TRAIN.TRAINED_DIR = osp.abspath(osp.join(__C.OUTPUT_DIR, 'trained_model'))

def _merge_a_into_b(a, b):
    '''Merge config dictionary a into config dictionary b, clobbering the 
    options in b whenever they are also specified in a
    '''
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the tpes must match, too
        print k
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype = b[k].dtype)
            else:
                raise ValueError(('Type mis match ({} vs. {}) '
                    'for config key: {}').format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise

        else:
            b[k] = v

def cfg_from_file(cfg_list):
    '''Load a config file and merge it into the default options.

    '''
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
