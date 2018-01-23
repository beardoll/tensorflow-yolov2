import cv2
from config.config import cfg
import os
import numpy as np
import glob
from utils.process import resize_image_keep_ratio, resize_label_keep_ratio
from dataset.imdb import imdb

class kitti(imdb):
    '''Pipeline for loading batch data for KITTI dataset.
    
    '''

    def __init__(self, image_set, data_argument = True):
        '''Initialize the class.

        Args:
            image_set: mark the property of data, value is 'train' or 'test'.    
        '''
        imdb.__init__(self, 'kitti_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, image_set)
        self._classes = ['Car', 'Pedestrian', 'Cyclist']
        self._dont_care = ['DontCare', 'Misc', 'Person_sitting', 'Truck', 'Tram']
        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._image_names, self._boxes, self._obj_num = self.read_annotations()
        self._shuffle_inds()
        
        # data_argument: if True, then apply data argument
        # screen: if True, means the data has been preprocessed, then
        #            the bounding boxes info are [cls_index, xc, yc, w, h], and have been
        #            normalized
        # image_ext: 'jpg' or 'png'
        self.config = {'data_argument': data_argument,
                         'screen'       : True,
                         'image_ext'    : '.png'
                      }
        
    def read_annotations(self):
        '''Read annotation files

        Returns:
            image_names: "xxxx.{image_ext}"
            boxes: normalized, [cls_idx, xc, yc, w, h]
        '''

        filelist_path = os.path.join(self._data_path, '*.txt')
        filelist = glob.glob(filelist_path)

        assert len(filelist) != 0, 'No annotationfiles in {} \
            !!'.format(filelist_path)

        image_names = []
        boxes = []
        obj_num = []
        for file_name in filelist:
            name = file_name.split('/')[-1]
            current_boxes = []
            with open(file_name, 'r') as file_to_read:
                object_num = 0
                line = file_to_read.readline()
                while line:
                    if object_num >= cfg.TRAIN.MAX_OBJ:
                        # We constrain the maximum object number
                        break

                    if self.config['screen'] == True:
                        cls, xc, yc, w, h = line.split()
                        cls = int(cls)
                        xc = float(xc)
                        yc = float(yc)
                        w = float(w)
                        h = float(h)
                        object_num += 1
                        current_boxes.append([cls, xc, yc, w, h])
                        line = file_to_read.readline()
                    else:
                        # Here we only support KITTI format
                        nn = os.path.join(self._data_path, name +
                                self.config['image_ext'])
                        assert os.path.exists(nn), 'Image path {} does not \
                            exists!'.format(nn)

                        img = cv2.imread(nn)
                        h, w, c = img.shape

                        object_name, truncation, occlusion, _, x1, y1, x2, y2, \
                                _, _ ,_, _, _, _, _ = line.split()
                        if object_name == 'Van':
                            object_name = 'Car'

                        if object_name in self._dont_care:
                            # We ignore the object with label 'DontCare'
                            line = file_to_read.readline()
                            continue
                        else:
                            truncation = float(truncation)
                            occlusion = int(occlusion)

                            x1 = float(x1)
                            y1 = float(y1)
                            x2 = float(x2)
                            y2 = float(y2)

                            cls = self._class_to_ind[object_name]
                            xcenter = (x1 + x2) * 1.0 / 2 / w
                            ycenter = (y1 + y2) * 1.0 / 2 / h

                            box_w = (x2 - x1) / w
                            box_h = (y2 - y1) / h

                            if truncation < 0.5 and occlusion < 3 and box_h > 5:
                                current_boxes.append([cls, xcenter, ycenter,
                                    box_w, box_h])
                                object_num += 1

                            line = file_to_read.readline()

                if object_num != 0:
                    name = name.split('.')[0]
                    image_names.append(name + self.config['image_ext'])
                    boxes.append(current_boxes)
                    obj_num.append(object_num)
        return image_names, boxes, obj_num

    def _shuffle_inds(self):
        '''Randomly permute the training data

        '''
        self._perm = np.random.permutation(range(len(self._image_names)))
        self._perm = np.array(self._perm, dtype=np.int)
        self._cur = 0

    def _get_next_batch_inds(self):
        if self._cur + cfg.TRAIN.BATCH >= len(self._image_names):
            self._shuffle_inds()

        inds = self._perm[self._cur: self._cur + cfg.TRAIN.BATCH]
        self._cur += cfg.TRAIN.BATCH

        #inds = xrange(4, cfg.TRAIN.BATCH+4)

        return inds

    def get_batch_data(self, w, h):
        '''Get batch data

        Args:
            w, h: The required image size

        Returns:
            images_array, labels_array: float32 ndarray
            obj_num_array: indicate the number of objects for each image
        '''
        inds = self._get_next_batch_inds()
        images_array = np.zeros((cfg.TRAIN.BATCH, h, w, 3))
        boxes_array = np.zeros((cfg.TRAIN.BATCH, cfg.TRAIN.MAX_OBJ, 5))
        obj_num_array = np.zeros(cfg.TRAIN.BATCH, dtype=np.int)
        i = 0
        retain = []
        for index in inds:
            name = self._image_names[index]
            name = os.path.join(self._data_path, name)
            assert os.path.exists(name), 'Image path {} does not exist!'\
                    .format(name)

            image = cv2.imread(name)
            org_h, org_w = image.shape[:2]

            # Transform BGR to RGB, and normalize the pixels
            cur_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cur_image = np.array(cur_image, dtype=np.float32)
            cur_image /= 255.0

            cur_boxes = self._boxes[index]
    
            if self.config['data_argument'] == True:
                cur_image, cur_boxes = self.process_image(cur_image, \
                        cur_boxes, w, h)
            else:
                cur_image = resize_image_keep_ratio(cur_image, w, h)
                cur_boxes = resize_label_keep_ratio(cur_boxes, org_w, 
                        org_h, w, h)

            images_array[i,:] = cur_image
            obj_num_array[i] = self._obj_num[index]


            boxes_array[i, 0:obj_num_array[i], :] = cur_boxes
            i = i + 1

        return images_array, labels_array, obj_num_array
