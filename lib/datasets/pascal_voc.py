import cv2
from config.config import cfg
import os
import numpy as np
import xml.etree.ElementTree as ET
from utils.process import resize_image_keep_ratio, resize_label_keep_ratio
from dataset.imdb import imdb

class pascal_voc(imdb):
    '''Pipeline for loading batch data for pascal voc dataset

    '''
    def __init__(self, image_set, year = 2007, data_argument = True):
        '''Initialize the class
        By default: Using 2007 dataset

        Args:
            image_set: mark the property of data, value is 'train' or 'test'
        '''
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._image_set = image_set
        self._classes = ('aeroplane', 'bicycle', 'bird', 'boat', 
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottdplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self._classes, xrange(len(self._classes))))
        self._data_path = os.path.join(cfg.DATA_DIR, 'VOCdevkit' + year,
                'VOC' + year)
        # image name (without ext)
        self._image_index = self._load_image_set_index()
        self._image_names, self._boxes, self._obj_num = self.read_annotations()
        self._shuffle_inds()

        # data_argument: if True, then implement data argument
        # image_ext: 'jpg' or 'png'
        self.config = {'data_argument' : data_argument,
                       'image_ext'     : '.jpg'
                      }

    def _load_image_set_index(self):
        '''Load the indexes listed in this dataset's image set file

        '''
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                self._image_set, '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index


    def read_annotations(self):
        '''Read annotation files

        Returns:
            image_names: "xxxx.{image_ext}"
            boxes: unnormalized, [cls_idx, xc, yc, w, h]
        '''
        image_names = []
        boxes = []
        obj_num = []
        for index in self.image_index:
            filename = os.path.join(self._data_path, 'Annotations', index +
                    '.xml')
            tree = ET.parse(filename)
            objs = tree.findall('object')
          
            # Exclude the samples labeled as difficult
            non_diff_objs = [obj for obj in objs if
                    int(obj.find('difficult').text) == 0]
            objs = non_diff_objs

            object_num = 0
            current_boxes = []
            for ix, obj in enumerate(objs):
                if object_num >= cfg.TRAIN.MAX_OBJ:
                    # We constrain the maximum object number
                    continue
                else:
                    bbox = obj.find('bndbox')
                    # Make pixel indexes 0-based
                    x1 = float(bbox.find('xmin').text) - 1
                    y1 = float(bbox.find('ymin').text) - 1
                    x2 = float(bbox.find('xmax').text) - 1
                    y2 = float(bbox.find('ymax').text) - 1
                    cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                    xc = (x1 + x2) / 2.0
                    yc = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    current_boxes.append([cls, xc, yc, w, h])
                    object_num += 1

            if object_num != 0:
                name = os.path.join(self._data_path, 'JPEGImages', index +
                    self._image_ext)
                image_names.append(name)
                boxes.append(current_boxes)
                obj_num.append(object_num)
        
        return image_names, boxes, obj_num

    def _get_gtbox_by_name(self, name):
        '''Get grounding truth box by name

        '''
        idx = self._image_names.index(name)
        return self._boxes[idx]

    def _shuffle_inds(self):
        '''Randomly permute the training data

        '''
        self._perm = np.random.permutation(range(self._image_names))
        self._perm = np.array(sef._perm, dtype = np.int)
        self._cur = 0

    def _get_next_batch_inds(self):
        if self._cur + cfg.TRAIN.BATCH >= len(self._image_names):
            self._shuffle_inds()
        
        inds = self._perm[self._cur: self._cur + cfg.TRAIN.BATCH]
        self._cur += cfg.TRAIN.BATCH

        return inds


    def get_batch_data(self, w, h):
        '''Get batch data

        Args:
            w, h: The required image size

        Returns:
            images_array, labels_array: float32 ndarray
            object_num_array: indicate the number of objects for each image
        '''
        inds = self._get_next_batch_inds()
        images_array = np.zeros((cfg.TRAIN.BATCH, h, w, 3))
        boxes_array = np.zeros((cfg.TRAIN.BATCH, cfg.TRAIN.MAX_OBJ, 5))
        obj_num_array = np.zeros(cfg.TRAIN.BATCH, dtype=np.int)

        for index in inds:
            name = self._image_names[index]
            assert os.path.exists(name), 'Image path {} does not exist!'\
                    .format(name)

            image = cv2.imread(name)
            org_h, org_w = image.shape[:2]

            # Transform BGR to RGB, and normalize the pixel value
            cur_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cur_image = np.array(cur_image, dtype=np.float32)
            cur_image /= 255.0

            # normalize the bbox
            cur_boxes = self._boxes[index]
            cur_boxes[1] /= org_w
            cur_boxes[2] /= org_h
            cur_boxes[3] /= org_w
            cur_boxes[4] /= org_h

            if self.config['data_argument'] == True:
                cur_image, cur_boxes = self.process_image(cur_image, \
                        cur_boxes, w, h)
            else:
                cur_image = resize_image_keep_ratio(cur_image, w, h)
                cur_boxes = resize_label_keep_ratio(cur_boxes, org_w, 
                        org_h, w,h)

            images_array[i,:] = cur_image
            obj_num_array[i] = self._obj_num[index]

            boxes_array[i, 0:obj_num_array[i], :] = cur_boxes

        return images_array, labels_array, obj_num_array
   
    def _get_results_filename(self, cls):
        '''Get path for file saving detection results
        Args:
            cls: classname for this file
        Returns:
            path
        '''
        filename = 'pascal_voc_' + cls + '.txt'
        path = os.path.join(cfg.TEST.OUTPUT_DIR, 'test', 'txt', filename)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


    def evaluate_detections(self, all_boxes):
        ''' Evaluate the performance of your model
    
        args:
            all_boxes[cls][image_name]: (xc, yc, w, h, confidence)
        '''
        for cls_ind, cls in enumerate(self._classes):
            filename = _get_results_filename(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_names):
                    dets = all_boxes[cls][index]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n', 
                                format(index, dets[k, -1], dets[k, 0], 
                                    dets[k, 1], dets[k, 2], dets[k, 3]))

        aps = []
        for cls_ind, cls in enumerate(self._classes):
            ap = self._evaluate_mAP(cls)
            print('AP for {} = {:.4f}'.format(cls, ap)) 
            aps += [ap]

        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def _evaluate_mAP(self, classname):
        '''Using pascal voc 11-points method to evalute mAP for certain class

        Should make sure that the medium text file has been generated
        '''
        filename = self._get_results_filename(classname)
        cls_index = self.classes.index(classname)
        
        # Construct class_recs: mainly record whether the object has been matched
        class_recs = {}
        npos = 0  # number of objects
        for image_name in self.image_names:
            gt_boxes = self._get_gtbox_by_name(image_name)
            bbox = []
            if gt_boxes is not None:
                for box in gt_boxes:
                    if box[0] == cls_index:
                        if bbox == []:
                            bbox = np.hstack((bbox, box[1:4]))
                        else:
                            bbox = np.vstack((bbox, box[1:4]))
                        npos += 1

            det = [False] * npos
            class_recs[image_name] = {'bbox': bbox,
                                      'det' : det}

        with open(filename, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:
            splitlines = [x.strip().split(' ') for x in lines]
            image_names = [x[0] for x in splitlines]
            confidence = [x[1] for x in splitlines]
            BB = np.array([[float(z) for z in x[2:]]] for x in splitlines)

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_names = image_names[sorted_ind, :]

            # Calculate tp and fp
            nd = len(image_names)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_names[d]]
                bb = BB[d,: ].astype(float)
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps and intersection
                    overlaps = []
                    for idx in BBGT.shape[0]:
                        current_gtbox = BBGT[idx, :]
                        overlaps.append(iou(current_gtbox, bb))

                    ovmax = np.max(overlaps)
                    ovmax_ind = np.argmax(overlaps)

                    if ovmax > ovthresh:
                        if not R['det'][ovmax_ind]:
                            tp[d] = 1
                            R['det'][ovmax_ind] = 1
                        else:
                            fp[d] = 1
                    else:
                        fp[d] = 1

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)

            # Avoid divide by zero in case the first detection matches a
            # difficult ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
        else:
            rec = -1
            prec = -1
            ap = -1

        return rec, prec, ap

    def _voc_ap(self, rec, prec, use_07_metric = True):
        '''Compute VOC AP given precision and recall
        
        if use_07_metric is true, uses the 
        VOC 07 11 point method (default: True)
        
        '''
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11
        else:
            # correct AP calculation
            # First append sentinel values at the end
            mrec = np.concatenate([0.], rec, [1.])
            mpre = np.concatenate([1.], prec, [0.])

            # Compute the precision envelop
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calcute area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            
        return ap









