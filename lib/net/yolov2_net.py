import tensorflow as tf
from net.network import Network
from config.config import cfg
import numpy as np
import copy

class YOLOv2_net(Network):
    def __init__(self, labels = None, seen = None, is_training = True, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.layers = dict({'data': self.data})
        self.trainable = trainable
        self.is_training = is_training
        self.num_outputs = cfg.TRAIN.BOX_NUM * (len(cfg.TRAIN.CLASSES) + 5)
        self.labels = labels
        self.seen = seen
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 32, 1, 1, name='conv1', batchnorm=True)
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 64, 1, 1, name='conv3', batchnorm=True)
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 128, 1, 1, name='conv5', batchnorm=True)
             .conv(1, 1, 64, 1, 1, name='conv6', batchnorm=True)
             .conv(3, 3, 128, 1, 1, name='conv7', batchnorm=True)
             .max_pool(2, 2, 2, 2, name='pool8')
             .conv(3, 3, 256, 1, 1, name='conv9', batchnorm=True)
             .conv(1, 1, 128, 1, 1, name='conv10', batchnorm=True)
             .conv(3, 3, 256, 1, 1, name='conv11', batchnorm=True)
             .max_pool(2, 2, 2, 2, name='pool12')
             .conv(3, 3, 512, 1, 1, name='conv13', batchnorm=True)
             .conv(1, 1, 256, 1, 1, name='conv14', batchnorm=True)
             .conv(3, 3, 512, 1, 1, name='conv15', batchnorm=True)
             .conv(1, 1, 256, 1, 1, name='conv16', batchnorm=True)
             .conv(3, 3, 512, 1, 1, name='conv17', batchnorm=True)
             .max_pool(2, 2, 2, 2, name='pool18')
             .conv(3, 3, 1024, 1, 1, name='conv19', batchnorm=True)
             .conv(1, 1, 512, 1, 1, name='conv20', batchnorm=True)
             .conv(3, 3, 1024, 1, 1, name='conv21', batchnorm=True)
             .conv(1, 1, 512, 1, 1, name='conv22', batchnorm=True)
             .conv(3, 3, 1024, 1, 1, name='conv23', batchnorm=True)
             .conv(3, 3, 1024, 1, 1, name='conv24', batchnorm=True)
             .conv(3, 3, 1024, 1, 1, name='conv25', batchnorm=True))
        
        (self.feed('conv17')
             .conv(1, 1, 64, 1, 1, name='conv26', batchnorm=True)
             .reorg(stride=2, name='reorg27'))

        (self.feed('conv25', 'reorg27')
             .concat(axis=3, name='concate28')
             .conv(3, 3, 1024, 1, 1, name='conv29', batchnorm=True)
             .conv(1, 1, self.num_outputs, 1, 1, name='conv30', 
                 activation='linear', batchnorm=False))

        if self.is_training == True:
            (self.feed('conv30')
                 .region(self.labels, self.seen, name='region31'))

