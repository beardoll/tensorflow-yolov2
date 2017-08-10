import tensorflow as tf
from net.network import Network
from config.config import cfg


class YOLOv2_net(Network):
    def __init__(self, is_training, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.layers = dict({'data': self.data})
        self.trainable = trainable
        self.is_training = is_training
        self.num_outputs = 5 * (len(cfg.TRAIN.CLASSES) + 5)
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
             .conv(1, 1, self.num_outputs, 1, 1, name='region30', 
                 activation='linear', batchnorm=False))

    def iou(self, boxes, query_box):
        '''Compute ious between boxes and query_box

        python function

        Args:
            boxes: N1 x 4 ndarray, with format (xc, yc, w, h)
            query_box: 1 x 4 ndarray, with format (xc, yc, w, h)
        Returns:
            overlaps: N1 x 1 ndarray of overlap between boxes and query_box 
        '''

        # Tranform (xc, yc, w, h) -> (x1, y1, x2, y2)
        boxes = np.concatenate(([boxes[:, 0] - boxes[:, 2] / 2, \
                                 boxes[:, 1] - boxes[:, 3] / 2, \
                                 boxes[:, 0] + boxes[:, 2] / 2, \
                                 boxes[:, 1] + boxes[:, 3] / 2]), axis=1)
        
        query_box = np.array(([query_box[0] - query_box[2] / 2, \
                               query_box[1] - query_box[3] / 2, \
                               query_box[0] + query_box[2] / 2, \
                               query_box[1] + query_box[3] / 2]),
                               dtype=np.float32)

        # Repeat query_box along y-axis
        temp = np.zeros((boxes.shape[0], 1), dtype=np.float32)
        query_box = query_box + temp

        # Calculate the left-up points and right-down points 
        # of overlap areas
        lu = boxes[:, 0:2] * (boxes[:, 0:2] >= query_box[:, 0:2]) + \
                query_box[:, 0:2] * (boxes[:, 0:2] < query_box[:, 0:2])

        rd = boxes[:, 2:4] * (boxes[:, 2:4] <= query_box[:, 2:4]) + \
                query_box[:, 2:4] * (boxes[:, 2:4] > query_box[:, 2:4])

        # itersection = (iter_r - iter_l) * (iter_d - iter_u)
        intersection = rd - lu

        inter_square = intersection[:, 0] * intersection[:, 1]

        # Elimated those itersection with w or h < 0
        mask = np.array(intersection[:, 0] > 0, np.float32) * \
               np.array(intersection[:, 1] > 0, np.float32)

        inter_square = mask * inter_square
        
        # Calculate the boxes square and query_box square
        square1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        square2 = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])

        return inter_square / (square1 * square2 - inter_square + 1e-6)

    def logistic(self, x):
        return 1./(1 + np.exp(-x))

    def restore_box(self, box, h, w, map_height, map_width, prior_h, prior_w):
        '''Restore box (cx, cy, w, h) from predict info (tx, ty, tw, th)

        Args:
            box: (tx, ty, tw, th)
            h: the index in y-axis of feature map
            w: the index in x-axis of feature map
            map_height, map_width: the size of feature map
            prior_h, prior_w: the prior size of anchor
        Returns:
            box: (xc, yc, w, h), normalized
        '''
        box_x = (w + self.logistic(box[0])) / map_width
        box_y = (h + self.logistic(box[1])) / map_height
        box_w = np.exp(box[2]) * prior_w / map_width
        box_h = np.exp(box[3]) * prior_h / map_height
        return np.array([box_x, box_y, box_w, box_h])

    def compute_coord_delta(self, box1, box2, h, w, map_height, map_width,
            prior_h, prior_w, scale):
        '''Compute the coords loss between box1 and box2

        Args:
            box1, box2: (xc, yc, w, h), box2 is gt_box
            h: the index in y-axis of feature map
            w: the index in x-axis of feature map
            map_height, map_width: the size of feature map
            prior_h, prior_w: the prior size of anchor
            scale: the coeffience for the square loss

        Returns:
            delta: (tx_delta, ty_delta, tw_delta, th_delta)
        '''
        
        # Transform (xc, yc, w, h) -> (log_tx, log_ty, tw, th)
        log_tx1 = box1[0] * map_width - w
        log_ty1 = box1[1] * map_height - h
        tw1 = np.log(box1[2] * map_width / prior_w)
        th1 = np.log(box1[3] * map_height / prior_h)
        
        log_tx2 = box2[0] * map_width - w
        log_ty2 = box2[1] * map_height - h
        tw2 = np.log(box2[2] * map_width / prior_w)
        th2 = np.log(box2[3] * map_height / prior_h)

        delta = np.zeros((4,1), dtype=np.float32)
        delta[0] = scale * (log_tx2 - log_tx1)
        delta[1] = scale * (log_ty2 - log_ty1)
        delta[2] = scale * (tw2 - tw1)
        delta[3] = scale * (th2 - th1)

        return delta
        

    def cal_loss_py(self, predicts, labels, seen):
        '''Calculate loss between predicts and labels

        python function, callable for tensorflow using tf.py_func

        Args:
            predicts: Commonly batch x map_height x map_width x boxes_info
                      boxes_info: info for several (commonly 5) boxes,
                      including coords(4), object(1), class_prob(class_num)
            labels: ground-truth bounding box, batch x num_objects x 
                    (cls(1), coords(4))
            seen: The number of pictures that have been fed into the network
        Returns:
            loss:  loss over of all data
        Warnings:
            Note that the real box coords in predicts can only be grained after
            applying function self.restore_box()
        '''
        
        batch_size = cfg.TRAIN.BATCH
        map_height = predicts.shape[1]
        map_width = predicts.shape[2]

        box_num = cfg.TRAIN.BOX_NUM
        box_info_len = 4 + 1 + len(cfg.TRAIN.CLASSES)
        
        # Four evaluation criterions
        recall = 0
        avg_iou = 0
        avg_obj = 0
        avg_cat = 0
        avg_anyobj = 0

        # Total objects in labels
        obj_count = 0

        for b in range(batch_size):
            label_this_batch = labels[b, :, :]
            predict_this_batch = predicts[b, :, :]
            delta = np.zeros((map_height, map_width, box_num * box_info_len),
                    dtype = np.float32)
            for h in range(map_height):
                for w in range(map_width):
                    for k in range(box_num):
                        box_info = predict_this_batch[h, w, 
                                k * box_info_len: (k+1) * box_info_len]

                        prior_w = cfg.TRAIN.ANCHORS[2*k]
                        prior_h = cfg.TRAIN.ANCHORS[2*k+1]
                        box = self.restore_box(box_info[0:4], h, w, map_height,
                                map_width, prior_h, prior_w)
                        
                        gt_boxes = np.array(label_this_batch[:, 1:5])
                        
                        # iou between current box and gt boxes
                        box_iou = self.iou(gt_boxes, box)

                        if box_iou > cfg.TRAIN.THRESHOLD:
                            # If the box iou exceed overlaps,
                            # then the loss is zero
                            delta[h, w,: k*box_info_len+4] = 0
                        else:
                            delta[h, w,: k*box_info_len+4] = \
                                cfg.TRAIN.NOOBJECT_SCALE * (0 -self.logistic(box_info[4]))

                        avg_anyobj += self.logistic(box_info[4])

                        if seen < 12800:
                            # In the early feeding for 12800 pictures,
                            # the coord loss for each box should be calculated.
                            # Here, the loss is the deviated from the prior
                            # anchor.
                            truth_box_x = (w + 0.5) / map_width
                            truth_box_y = (h + 0.5) / map_height
                            truth_box_w = prior_w / map_width
                            truth_box_h = prior_h / map_height
                            truth_box = np.array([truth_box_x, truth_box_y,
                                                  truth_box_w, truth_box_h])

                            delta[h, w,: k*box_info_len, k*box_info_len+4] = \
                                    self.compute_coord_delta(box, truth_box, \
                                    h, w, map_height, map_width, prior_h, prior_w, 0.01)

            label_num = label_this_batch.shape[0]
            obj_count += label_num
            for m in range(label_num):
                """
                For each gt_box, we find one responsible pred box,
                and compute coord loss, class loss for that pred box
                """
                current_label = label_this_batch[m,: ]
                truth_box = current_label[1:5]
                # Fine the pixel index w.r.t feature map
                w = int(truth_box[0] * map_width)
                h = int(truth_box[1] * map_height)
                best_iou = 0
                best_idx = 0

                prior_w = cfg.TRAIN.ANCHORS[2*k]
                prior_h = cfg.TRAIN.ANCHORS[2*k+1]
                
                
                for k in range(box_num):
                    box_info = predict_this_batch[h, w, k * box_info_len: (k+1)
                            * box_info_len]

                    box = self.restore_box(box_info[0:4], h, w, map_height, 
                                map_width, prior_h, prior_w)

                    
                    # We make the centroids of truth_box and box the same
                    truth_shift = truth_box
                    truth_shift[0] = 0.0
                    truth_shift[1] = 0.0
                    truth_shift = truth_shift[np.newaxis,:]

                    box_iou = self.iou(truth_shift, box)
                    if box_iou > best_iou:
                        best_iou = box_iou
                        best_idx = k

                best_box_info = predict_this_batch[h, w, best_idx *
                        box_info_len: (best_idx+1) * box_info_len]

                best_box = self.restore_box(best_box_info[0:4], h, w,
                        map_height, map_width, prior_h, prior_w)

                # Recalculate iou
                truth_shift = truth_box[np.newaxis,: ]
                best_iou = self.iou(truth_shift, best_box)

                if best_iou > 0.5:
                    recall += 1

                avg_iou += best_iou

                avg_obj += self.logistic(best_box_info[4])

                # Coords loss
                delta[h, w, best_idx * box_info_len: best_idx * box_info_len +
                        4] = self.compute_coord_delta(best_box, truth_box, \
                                h, w, map_height, map_width, prior_h, prior_w,\
                                cfg.TRAIN.COORD_SCALE * (2 - truth_box[2]*truth_Box[3]))

                # Object loss
                delta[h, w, best_idx * box_info_len + 4] = \
                        cfg.TRAIN.OBJECT_SCALE * (best_iou - self.logistic(best_box_info[4]))

                # class prob loss
                cls = truth_box[0]
                temp = np.zeros(len(cfg.TRAIN.CLASSES), dtype = np.float32)
                temp[cls] = 1.0

                delta[h, w, best_idx * box_info_len + 5: (best_idx + 1) *
                        box_info_len] = cfg.TRAIN.CLASS_SCALE * (temp - \
                                best_box_info[5:])

                avg_cat += best_box_info[5+best_idx]

        print("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall:\
                %f, count %d"%(avg_iou/obj_count, avt_cat/obj_count,
                    avg_anyobj/(map_width*map_height*self.num_outputs*batch),
                    recall/obj_count, count))


        return np.square(delta)



    def loss(self, predicts, labels, seen):
        '''Calculate the training loss

        The explanation of args please see function "self.cal_loss_py"
        All args here are tensorflow objects
        '''

        loss = tf.py_func(self.cal_loss_py, [predicts, labels, seen],
                [tf.float32])

        loss = tf.convert_to_tensor(loss)
        
        return loss

