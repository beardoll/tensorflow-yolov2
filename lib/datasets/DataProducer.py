import cv2
from config.config import cfg
import os
import numpy as np
import glob
from utils.process import resize_image_keep_ratio, resize_label_keep_ratio

class DataProducer(object):
    '''Pipeline for loading batch data.
    
    Here, we design some data argumentation operations.
    These operations are used to enrich the dataset.
    Note that the labels will be processed too.
    '''

    def __init__(self, image_set, data_argument = True):
        '''Initialize DataProducer.

        By default the images and corresponding annotations are saved under the
        same directory.

        Args:
            image_set: mark the property of data, value is 'train' or 'test'.    
        '''
        
        '''self.__config
        keys:
            data_argument: if True, then apply data argument
            screen: if True, means the data has been preprocessed, then
                    the bounding boxes info are [cls_index, xc, yc, w, h], and have been
                    normalized
            image_ext: 'jpg' or 'png'
        '''
        self.__config = {'data_argument': True,
                         'screen'       : True,
                         'image_ext'    : '.png'
                         }
        
        self.__image_set = image_set
        # Path for images and annotations
        self.__data_path = os.path.join(cfg.DATA_DIR, image_set)
        self.__image_names, self.__boxes, self.__obj_num = self.read_annotations()
        self.__classes = cfg.TRAIN.CLASSES
        self.__class_to_ind = dict(zip(self.__classes,
            xrange(len(self.__classes))))
        self.__shuffle_inds()

        
    def read_annotations(self):
        '''Read annotation files

        Returns:
            image_names: "xxxx.{image_ext}"
            boxes: normalized, [cls_idx, xc, yc, w, h]
        '''

        filelist_path = os.path.join(self.__data_path, '*.txt')
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

                    if self.__config['screen'] == True:
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
                        nn = os.path.join(self.__data_path, name +
                                self.__config['image_ext'])
                        assert os.path.exists(nn), 'Image path {} does not \
                            exists!'.format(nn)

                        img = cv2.imread(nn)
                        h, w, c = img.shape

                        object_name, truncation, occlusion, _, x1, y1, x2, y2, \
                                _, _ ,_, _, _, _, _ = line.split()
                        if object_name == 'Van':
                            object_name = 'Car'

                        if object_name in cfg.TRAIN.DONT_CARE:
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

                            cls = self.__class_to_ind[object_name]
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
                    image_names.append(name + self.__config['image_ext'])
                    boxes.append(current_boxes)
                    obj_num.append(object_num)
        return image_names, boxes, obj_num

    def __shuffle_inds(self):
        '''Randomly permute the training data

        '''
        self.__perm = np.random.permutation(range(len(self.__image_names)))
        self.__perm = np.array(self.__perm, dtype=np.int)
        self.__cur = 0

    def __get_next_batch_inds(self):
        if self.__cur + cfg.TRAIN.BATCH >= len(self.__image_names):
            self.__shuffle_inds()

        inds = self.__perm[self.__cur: self.__cur + cfg.TRAIN.BATCH]
        self.__cur += cfg.TRAIN.BATCH

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
        inds = self.__get_next_batch_inds()
        images_array = np.zeros((cfg.TRAIN.BATCH, h, w, 3))
        boxes_array = np.zeros((cfg.TRAIN.BATCH, cfg.TRAIN.MAX_OBJ, 5))
        obj_num_array = np.zeros(cfg.TRAIN.BATCH, dtype=np.int)
        i = 0
        retain = []
        for index in inds:
            name = self.__image_names[index]
            name = os.path.join(self.__data_path, name)
            assert os.path.exists(name), 'Image path {} does not exist!'\
                    .format(name)

            image = cv2.imread(name)
            org_h, org_w = image.shape[:2]

            # Transform BGR to RGB, and normalize the pixels
            cur_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cur_image = np.array(cur_image, dtype=np.float32)
            cur_image /= 255.0

            cur_boxes = self.__boxes[index]
    
            if self.__config['data_argument'] == True:
                cur_image, cur_boxes = self.process_image(cur_image, \
                        cur_boxes, w, h)
            else:
                cur_image = resize_image_keep_ratio(cur_image, w, h)
                cur_boxes = resize_label_keep_ratio(cur_boxes, org_w, 
                        org_h, w, h)

            images_array[i,:] = cur_image
            obj_num_array[i] = self.__obj_num[index]


            boxes_array[i, 0:obj_num_array[i], :] = cur_boxes
            i = i + 1

        return images_array, boxes_array, obj_num_array

    def test_process(self, image_path, boxes):
        '''Test the preprocess for image and boxes
        
        Args:
            image_path: the name of image, by default the test image is stored
                        in current folder
            boxes: the bounding boxes, with format [cls, x1, x2, y1, y2]
                   x, y are the real coords
        '''
        print "Test the preprocess for image and boxes!"

        image = cv2.imread(image_path)
        
        org_h, org_w, org_c = image.shape
        print "origin height: {}, origin_width: {}, origin_channels: {}".\
                format(org_h, org_w, org_c)
        
        org = image
        for box in boxes:
            x1 = box[1]
            y1 = box[2]
            x2 = box[3]
            y2 = box[4]
            
            # Write the bounding box to origin image
            cv2.rectangle(org, (int(x1), int(y1)), (int(x2),
                int(y2)), (0, 255, 0), 2)
            xc = (x1+x2)/2/org_w
            yc = (y1+y2)/2/org_h
            bw = (x2-x1)/org_w
            bh = (y2-y1)/org_h
            box[1:] = [xc, yc, bw, bh]
        
        print "The origin image with bounding box is written into 'org.png'"
        cv2.imwrite("org.png", org)

        # Transform BGR to RGB, and normalize the pixels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)
        image /= 255.0

        print "Now preprocess the origin image, resize to (600, 400)"
        processed_image, processed_boxes = self.process_image(image, boxes, 600, 400)

        # The pixels of preprocessed image are in range [0, 1]
        processed_image *= 255
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        for box in processed_boxes:
            xc = box[1]
            yc = box[2]
            bw = box[3]
            bh = box[4]

            # The preprocessed image is of size (600, 400)
            x1 = (xc - bw/2) * 600
            y1 = (yc - bh/2) * 400
            x2 = (xc + bw/2) * 600
            y2 = (yc + bh/2) * 400
            
            # Write bounding boxes to preprocessed image
            cv2.rectangle(processed_image, (int(x1), int(y1)), (int(x2),
                int(y2)), (0, 255, 0), 2)

        print "The preprocessed image is stored in 'test.png'"
        
        cv2.imwrite("test.png", processed_image)


    def process_image(self, image, boxes, sized_w, sized_h):
        '''Process one image.

        We will resize the image, deviate the image from original relative
        position, change the hue, saturation, exposure of the image
        The boxes will be resized and deviated too.

        Args:
            image: the original image(RGB format 3-channel matrix, normalized
                   to range [0, 1]).
            boxes: [class_idx, xcenter, ycenter, w, h]
                   "class_idx" is used to mark the class of the boxed object.
                   "xcenter, ycenter, w, h" are normalized to range [0, 1]
                   according to the original with and height of image.
            sized_w, sized_h: the resized shape of image.

        Returns:
            processed_image: the image after processed.
            processed_boxes: the boxes after processed.

        '''
        # The original image size.
        origin_h, origin_w, origin_c = image.shape

        # The extend of jitter.
        jitter = cfg.TRAIN.JITTER

        # Hue, saturation and exposure.
        hue = cfg.TRAIN.HUE
        saturation = cfg.TRAIN.SATURATION
        exposure = cfg.TRAIN.EXPOSURE

        # The output image.
        processed_image = 0.5 * np.ones((sized_h, sized_w, origin_c),
                dtype=np.float32)

        # The deviated value in width and height.
        dw = jitter * origin_w
        dh = jitter * origin_h

        # The new width/height ratio after jitter (deviated).
        new_ratio = (origin_w + np.random.uniform(-dw, dw, size = 1)) /\
                    (origin_h + np.random.uniform(-dh, dh, size = 1))

        # Scale the processed_image.
        scale = np.random.uniform(0.25, 2, size = 1)

        # Adjust (sized_h, sized_w) -> (nh, nw).
        # If (nh, hw) is larger than (sized_h, sized_w),
        # then the out-of-bound region in (nh, nw) will be discarded
        # when putting (nh, nw) pixels into processed_image.
        nw = 0
        nh = 0
        if(new_ratio < 1):
            # h > w
            nh = scale * sized_h
            nw = nh * new_ratio
        else:
            # h <= w
            nw = scale * sized_w
            nh = nw / new_ratio
        
        nw = int(nw)
        nh = int(nh)

        # The deviated value between (sized_h, sized_w) and (nh, hw).
        # It's used when putting (nh, nw) pixels into processed_image.
        dx = np.random.uniform(0, sized_w - nw, size = 1)
        dy = np.random.uniform(0, sized_h - nh, size = 1)
        dx = int(dx)
        dy = int(dy)

        # Resize the origin image to (nh, nw) size
        sized_image = cv2.resize(image, (nw, nh))

        # Suppose nh<sized_h and nw<sized_w, if the deviated value dy,dx are zero,
        # then the smaller image (nh, nw) will be put in the left-top corner
        # of processed_image. So "dy, dx" decide the offset within image 
        # (sized_h, sized_w).
        # However, if nw > sized_w, then the out-of-bound pixels will be
        # discarded, and "dx, dy" decide which part of (nh, nw) will be
        # injected into (sized_h, sized_w).

        processed_image[max(dy, 0):min(nh+dy, sized_h), \
                max(dx, 0):min(nw+dx, sized_w),: ] \
            = sized_image[max(-dy, 0):min(-dy + sized_h, nh), \
                    max(-dx, 0):min(-dx + sized_w, nw),:]

        # The probability for applying flipping is 0.5
        # Here we only apply left-right flip.
        flip = np.random.randint(2)
        if flip == 1:
            # Flip for each channel
            for c in range(origin_c):
                processed_image[:, :, c] = np.fliplr(processed_image[:, :, c])

        
        # Change hue, saturation and exposure
        dhue = np.random.uniform(-hue, hue)
        dsat = np.random.uniform(1, saturation)
        if np.random.randint(2) == 1:
            dsat = 1.0 / dsat
        dexp = np.random.uniform(1, exposure)
        if np.random.randint(2) == 1:
            dexp = 1.0 / dexp

        # Before trainsform from RGB to HSV
        # The pixels should be restored to 0-255
        processed_image *= 255
        
        HSV = cv2.cvtColor(processed_image, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(HSV)
        
        # Normalized H to range [0, 1]
        H /= 360.0

        S = S * dsat
        V = V * dexp
        H = H + dhue
        
        H[np.where(H > 1)] -= 1
        H[np.where(H < 0)] += 1

        # Restore H to range [0, 360]
        H *= 360.0

        H = H[:, :, np.newaxis]
        S = S[:, :, np.newaxis]
        V = V[:, :, np.newaxis]

        # Distorted_image: (H, S, V) space image
        distorted_image = np.dstack((H, S, V))

        
        # Restore RGB format data from HSV format data
        # The pixels of restored image are of range [0, 255]
        processed_image = cv2.cvtColor(distorted_image, cv2.COLOR_HSV2RGB)
       
        # Normalize pixels in preprocessed image
        # Also bounding the pixels
        processed_image /= 255.0
        processed_image[np.where(processed_image < 0)] = 0
        processed_image[np.where(processed_image > 1)] = 1

        # Correct the boxes
        # Concretely, rescale and deviate
        box_x_delta = -dx * 1.0 / sized_w
        box_y_delta = -dy * 1.0 / sized_h
        box_x_scale = nw * 1.0 / sized_w
        box_y_scale = nh * 1.0 / sized_h

        processed_boxes = []
        for box in boxes:
            box_left   = box[1] - box[3] / 2.0
            box_right  = box[1] + box[3] / 2.0
            box_top    = box[2] - box[4] / 2.0
            box_bottom = box[2] + box[4] / 2.0
           

            box_left   = box_left  * box_x_scale - box_x_delta
            box_right  = box_right * box_x_scale - box_x_delta
            box_top    = box_top  * box_y_scale - box_y_delta
            box_bottom = box_bottom * box_y_scale - box_y_delta

            if flip == 1:
                # If left-right flipping
                swap = box_left
                box_left  = 1 - box_right
                box_right = 1 - swap

            # Bound the box
            box_left   = max(0, box_left)
            box_left   = min(1, box_left)
            box_right  = max(0, box_right)
            box_right  = min(1, box_right)
            box_top    = max(0, box_top)
            box_top    = min(1, box_top)
            box_bottom = max(0, box_bottom)
            box_bottom = min(1, box_bottom)

            # Get (xcenter, ycenter, box_width, box_height) which is normalized
            box_xcenter = (box_left + box_right) / 2
            box_ycenter = (box_top + box_bottom) / 2
            box_width   = box_right - box_left
            box_height  = box_bottom - box_top

            # Split those boxes with 0 length
            # .............

            processed_boxes.append([box[0], box_xcenter, box_ycenter,
                box_width, box_height])

        return processed_image, processed_boxes

if __name__ == '__main__':
    dp = DataProducer("test")
    boxes = []
    boxes.append([0, 981.58, 165.78, 1241.0, 374.0])
    boxes.append([0, 0.0, 151.2, 311.0, 335.5])
    boxes.append([0, 16.8, 163.5, 317.8, 300.7])
    dp.test_process("000831.png", boxes)
