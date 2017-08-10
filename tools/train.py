import tensorflow as tf
import _init_paths
from net.yolov2_net import YOLOv2_net
import os
from config.config import cfg

def train():
    pretrained_model = os.path.join(cfg.PRETRAINED_DIR, 'npy', 'yolov2.npy')
    assert os.path.exists(pretrained_model), \
            'Model path {} does not exist!'.format(pretrained_model)

    net = YOLOv2_net(is_training=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True,
        gpu_options = gpu_options))


    sess.run(tf.global_variables_initializer())

    print('Loading pretrained model from {}').format(pretrained_model)
    net.load(pretrained_model, sess)
    sess.close()


if __name__ == '__main__':
    train()
