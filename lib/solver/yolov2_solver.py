import tensorflow as tf
from config.config import cfg
import numpy as np
import os
from datasets.DataProducer import DataProducer
from utils.timer import Timer
from net.yolov2_net import YOLOv2_net
from utils.multigpu import average_gradients

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


    def feed_all_gpu(self, models, payload_per_gpu, batch_x, batch_y, seen):
        '''Construct feed_dict for each gpu

        Args:
            models: the first two elements are placeholder for inputs and
                    labels, 3th element is for 'seen' element, 3-5th 
                    elements are predicts, loss, grads
            payload_per_gpu: the number of inputs for each gpu
            batch_x, batch_y: a large batch of data (input and labels), which
                              will be distributed into each gpu
            seen: the amount of data that has been trained

        Returns:
            inp_dict: dictionary for input (feed dictionary)
        '''
        inp_dict = {}
        for i in range(len(models)):
            x, y, s, _, _ = models[i]
            start_pos = i * payload_per_gpu
            end_pos = (i+1) * payload_per_gpu
            inp_dict[x] = batch_x[start_pos: end_pos]
            inp_dict[y] = batch_y[start_pos: end_pos]
            inp_dict[s] = seen

        return inp_dict


    def train_net(self):
        '''Training yolov2 net here

        We use the theorem of multi-gpu training to divide the large batch to
        serveral small batches, and average the gradients of them in each
        iteration to updata the parameters
        '''
       
        # Divide a large batch of data into serveral small batches
        subdivision = cfg.TRAIN.SUBDIVISION 

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
            gpu_options = gpu_options))

        # Producer for data
        data_producer = DataProducer(self.__image_set)
        
        # Momentum for SGD
        momentum = cfg.TRAIN.MOMENTUM
       
        # Timer
        timer = Timer()

        with sess.as_default(), tf.device('/cpu:0'):
            learning_rate = tf.placeholder(tf.float32, shape=())
            opt = tf.train.MomentumOptimizer(learning_rate, momentum)
            models = []

            # Pretrained variables list
            pretrained_list = []

            for division_id in range(subdivision):
                with tf.device('/gpu: %d' %division_id):
                    with tf.name_scope('tower_%d' %division_id):
                        with tf.variable_scope("", reuse=division_id > 0):
                            y = tf.placeholder(tf.float32, shape=(None,
                                cfg.TRAIN.MAX_OBJ, 5))
                            seen = tf.placeholder(tf.int32, shape=())
                            net = YOLOv2_net(y, seen)

                            if division_id == 0:
                                self.__net = net   # For saving snapshots
                                # Only load pretrained model once
                                if self.__pretrained_model is not None:
                                    print('Loading pretrained model weights from{:s}')\
                                                    .format(self.__pretrained_model)
                                    net.load(self.__pretrained_model, sess)
                                    pretrained_list = net.pretrained_variable_list
                
                            deltas = net.get_output('region31')
                            total_loss = tf.reduce_mean(tf.reduce_sum(0.5 *
                                tf.square(deltas), axis= [1, 2, 3]))
                            #total_loss += tf.add_n(tf.get_collection('weight_decay')) /\
                            #              (division_id + 1)
                            
                            grads = opt.compute_gradients(total_loss)
                            models.append((net.data, y, seen, total_loss, grads))

            tower_x, tower_y, tower_seen, tower_losses, tower_grads = zip(*models)

            avg_loss_op = tf.reduce_mean(tower_losses)
            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))

            payload_per_gpu = int(cfg.TRAIN.BATCH / subdivision)

            sess.run(tf.variables_initializer(set(tf.global_variables()) -
                set(pretrained_list)))

            print('Finish initialization')

            self.saver = tf.train.Saver(max_to_keep=100)
            lr = cfg.TRAIN.LEARNING_RATE / cfg.TRAIN.BATCH
            seen = 0
            for batch_idx in range(cfg.TRAIN.MAX_ITERS):
                dim = 416
                if batch_idx % 10 == 0:
                    # Change the resolution of input images
                    dim = int((np.random.randint(10) + 10) * 32)
                    if batch_idx >= cfg.TRAIN.ITER_THRESH:
                        # In the rest procedure of training, we use the 
                        # maximum resolution 608 x 608
                        dim = 608

                    print 'Resize the image to %d x %d'%(dim, dim)


                images, labels, _ = data_producer.get_batch_data(dim, dim)
                
                images = np.array(images, dtype = np.float32)
                labels = np.array(labels, dtype = np.float32)

                if batch_idx in cfg.STEP_SIZE:
                    pos = cfg.STEP_SIZE.index(batch_idx)
                    lr *= cfg.SCALES[pos]

                inp_dict = self.feed_all_gpu(models, payload_per_gpu,
                        images, labels, seen)
                inp_dict[learning_rate] = lr

                timer.tic()
                _, _loss = sess.run([apply_gradient_op, avg_loss_op], inp_dict)
                timer.toc()

                print 'Iteration: %d, train loss is: %f, learning rate is: %f, time consume: %f'\
                        %(batch_idx, _loss, lr, timer.average_time)

                seen += cfg.TRAIN.BATCH

                if (batch_idx + 1) % cfg.TRAIN.SNAPSHOTS == 0:
                    self.snapshot(sess, batch_idx)

        sess.close()

