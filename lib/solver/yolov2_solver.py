import tensorflow as tf
from config.config import cfg
import numpy as np
import os
from datasets.DataProducer import DataProducer
from utils.timer import Timer
from net.yolov2_net import YOLOv2_net

class SolverWrapper(object):
    '''Solver wrapper for training and testing yolov2 network

    Snapshot files will be generated and stored
    '''

    def __init__(self, image_set, pretrained_model = None):
        '''Initialize solver wrapper

        Args:
            network: yolo network instance
            image_set: 'train' or 'test'
            pretrained_model: file format of '.npy' or '.ckpt'
        '''

        self.__pretrained_model = pretrained_model
        self.__image_set = image_set
        self.__output_dir = cfg.TRAIN.TRAINED_DIR

    def snapshot(self, sess, iter):
        '''Take a snapshot of the network

        Args:
            sess: session of tensorflow
            iter: the current global step
        '''
        net = self.__net
        
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)

        # The saved filename
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX !=
                '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(cfg.TRAIN.TRAINED_DIR, filename)

        self.saver.save(sess, filename, global_step = iter + 1)
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_net(self):
        '''Training yolov2 net here

        Including initialize network and loading data from DataProducer
        '''
        
        # Specify the usage of gpu memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
            gpu_options = gpu_options))

        with sess.as_default():
            data_producer = DataProducer(self.__image_set)
            
            labels_placeholder = tf.placeholder(tf.float32, shape=(None,
                cfg.TRAIN.MAX_OBJ, 5))
            seen_placeholder = tf.placeholder(tf.int32, shape=())

            self.__net = YOLOv2_net(labels_placeholder, seen_placeholder, True)

            deltas = self.__net.get_output('region31')
            total_loss = tf.reduce_mean(tf.reduce_sum(0.5 * tf.square(deltas),
                axis=[1, 2, 3]))

            #total_loss = tf.reduce_sum(0.5 * tf.square(deltas))
            
                #total_loss += tf.add_n(tf.get_collection('losses'))

            lr_placeholder = tf.placeholder(tf.float32, shape=())

            global_step = tf.Variable(0, trainable=False)

            momentum = cfg.TRAIN.MOMENTUM

            train_op = tf.train.MomentumOptimizer(lr_placeholder,
                    momentum).minimize(total_loss, global_step = global_step)

            sess.run(tf.global_variables_initializer())

            # Load pretrained variables
            if self.__pretrained_model is not None:
                print('Loading pretrained model weights from \
                    {:s}').format(self.__pretrained_model)

                self.__net.load(self.__pretrained_model, sess)

            #summary_op = tf.summary.merge_all()
            #summary_writer = tf.summary.FileWriter(cfg.TRAIN.SUMMARY_DIR,
            #        sess.graph)
            
            timer = Timer()

            self.saver = tf.train.Saver(max_to_keep=100)

            seen = 0

            lr = 0.0001
            for iteration in range(cfg.TRAIN.MAX_ITERS):
                if iteration%10 == 0:
                    # Change the resolution of input images
                    dim = int((np.random.randint(10) + 10) * 32)
                    if iter >= 80000:
                        # In the rest procedure of training, we use the
                        # maximum resolution 608 x 608
                        dim = 608
                    
                    #dim = 416
                    print "Resize image to: %d x %d"%(dim, dim)

                images, labels, obj_num = data_producer.get_batch_data(dim, dim)

                images = np.array(images, dtype = np.float32)
                labels = np.array(labels, dtype = np.float32)

                if iteration in cfg.STEP_SIZE:
                    pos = cfg.STEP_SIZE.index(iteration)
                    lr *= cfg.SCALES[pos]

                print '\n'
                print "Iteration: %d, doing training..."%(iteration+1)
                
                timer.tic()
                total_loss2, _ = sess.run([total_loss, train_op], feed_dict
                        = {self.__net.data: images, labels_placeholder:
                            labels, seen_placeholder:seen,
                            lr_placeholder: lr})

                timer.toc()

                print "The training loss is %f, lr: %f, time comsuming: %f"%(total_loss2, lr, timer.average_time)


                #if (iteration+1) % 100 == 1:
                #    # Write summary for each 100 iterations
                #    summary_str = sess.run(summary_op,
                #            feed_dict={self.__net.data: images,
                #                labels_placeholder: labels,
                #                seen_placeholder:seen})

                if (iteration+1) % cfg.TRAIN.SNAPSHOTS == 0:
                    self.snapshot(sess, iteration)

                seen += cfg.TRAIN.BATCH

            sess.close()

