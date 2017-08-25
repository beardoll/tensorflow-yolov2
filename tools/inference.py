import _init_paths
import tensorflow as tf
import numpy as np
from net.yolov2_net import YOLOv2_net
from config.config import cfg
from utils.process import softmax, logistic, iou, resize_image_keep_ratio
import cv2

# The network size, according to the input size of final hundreds of training
NET_WIDTH = 416
NET_HEIGHT = 416

def restore_boxes(boxes, map_w, map_h):
    '''Restore the boxes from (tx, ty, tw, th) to (xc, yc, w, h)

    The number of boxes are box_num * map_w * map_h

    Args:
        boxes: boxes with complete informations, the coords are (tx, ty, tw, th)
        map_w, map_h: the width and height of the feature map, they are used to
           restore the coords in boxes, because each box is w.r.t specified
           grid in the feature maps
    '''
    box_num = cfg.TRAIN.BOX_NUM
    for cls, box in enumerate(boxes):
        n = cls
        box_idx = n % box_num  # the box index within [0, box_num]
        n /= box_num
        w = n % map_w          # the index of x-axis of box
        n /= map_w
        h = n % map_h          # the index of y-axis of box

        # Extract prior size of box according to box_idx
        prior_w = cfg.TRAIN.ANCHORS[2*box_idx]
        prior_h = cfg.TRAIN.ANCHORS[2*box_idx + 1]
        
        box_x = (logistic(box[0]) + w) / map_w
        box_y = (logistic(box[1]) + h) / map_h
        box_w = (np.exp(box[2]) * prior_w) / map_w
        box_h = (np.exp(box[3]) * prior_h) / map_h

        #print box_x, box_y, box_w, box_h
        boxes[cls, 0:4] = [box_x, box_y, box_w, box_h]

    return boxes


def filter_boxes(boxes, threshold):
    '''Get delegate box in boxes

    confidence = max(prob) * Ptr(obj)
    Only those confidence > threshold will be recorded

    Args:
        boxes: coords(4) + obj(1) + class_num
    
    Returns:
        BB: dict, with 'box', 'confidence', 'class'
    '''
    
    box_num = cfg.TRAIN.BOX_NUM

    BB = []
    for box in boxes:
        obj = logistic(box[4])

        logits = box[5:]
        probs = softmax(logits)
        max_prob = np.max(probs)
        confidence = obj * max_prob
        temp = {} 
        if confidence > threshold:
            temp['box'] = box[0:4]
            temp['confidence'] = confidence
            temp['class'] = np.argmax(probs)
            BB.append(temp)

    return BB

def nms(BB, nms_threshold):
    '''Apply nms for boxes in BB

    Once a box is specified for detecting an object, those boxes with iou
    larger than 'nms_threshold' will be discarded.

    Args:
        BB: dict, with member 'box', 'confidence' and 'class'
        nms_threshold: the maximum overlaps between two gt_boxes
    
    Returns:
        survived_BB: the filtered BB
    '''

    confidence = np.array([float(x['confidence']) for x in BB])

    # Descendantly sort confidence
    sorted_inds = np.argsort(-confidence)
    sorted_scores = np.argsort(-confidence)

    survived_BB = []

    for i in range(len(sorted_inds)):
        b1 = BB[sorted_inds[i]]
        if b1['confidence'] == 0:
            continue
        survived_BB.append(b1)
        for j in xrange(i+1, len(sorted_inds)):
            b2 = BB[sorted_inds[j]]
            if (iou(b1['box'], b2['box']) > nms_threshold):
                b2['confidence'] = 0

    return survived_BB


def draw_results(nms_BB, org_img, net_w, net_h):
    '''Draw bbox of nms_BB in org_img

    Note that the bbox has been resized to (net_h, net_w)
    We should rescale bbox to size of org_img
    If we don't keep the ratio of width/height of org_img in the process of
    resize, maybe it's unneccesary to use 'net_w' and 'net_h'
    
    Args:
        nms_BB: dict, with member 'box', 'confidence', 'cls'
        org_img: the original image
        new_w, net_h: the original image has been reshaped to (new_w, net_h)
                      and feed into the network 

    Returns:
        org_img: image with bbox and class info
    '''

    org_h, org_w = org_img.shape[:2]
    classes = cfg.TRAIN.CLASSES
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    for bb in nms_BB:
        bbox = bb['box']
        cls = bb['class']
        confidence = bb['confidence']
        class_name = classes[cls]
        rescale_factor = np.array([org_w, org_h, org_w, org_h])
        rescale_bbox = bbox * rescale_factor
        x1 = rescale_bbox[0] - rescale_bbox[2] / 2.0
        y1 = rescale_bbox[1] - rescale_bbox[3] / 2.0
        x2 = rescale_bbox[0] + rescale_bbox[2] / 2.0
        y2 = rescale_bbox[1] + rescale_bbox[3] / 2.0

        #print rescale_bbox
        #print x1, y1, x2, y2
        print int(x1), int(y1), int(x2), int(y2)
        print confidence

        org_img = cv2.rectangle(org_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255,
            0), 2)
        #cv2.putText(org_img, class_name + " " + str(confidence), (int(x1),
        #    int(y1)-10), font, 0.75, (0, 255, 0), 1)

    return org_img

def correct_boxes(boxes, org_w, org_h, net_w, net_h):
    '''Correct boxes from w.r.t the net size -> w.r.t the original size

    Note that the origin image has been resize to fit the input size of the
    network while retaining the width/height ratio. And the content of image
    has been put at the center of the resized image, and the two sides of
    resized image are grey pixels (background)

    Args:
        boxes: 2-dimension ndarrays
        org_w, org_h: the original size of image
        net_w, net_H: the size of inputs for network
    '''

    nw = 0
    nh = 0
    ratio = float(net_w / org_w) / float(net_h / org_h)
    if ratio < 1:
        # The resize ratio for width is larger than height
        nw = net_w
        nh = int(org_h * net_w / org_w)
    else:
        # The resize ratio for height is larger than width
        nh = net_h
        nw = int(org_w * net_h / org_h)

    for box in boxes:
        box[0] = (box[0] - (net_w - nw) / 2.0 / net_w) / (nw*1.0/net_w)
        box[1] = (box[1] - (net_h - nh) / 2.0 / net_h) / (nh*1.0/net_h)
        box[2] *= net_w * 1.0 / nw
        box[3] *= net_h * 1.0 / nh
    
    return boxes


def detect(input_image, output_image, threshold):
    '''Forward input_image to the yolov2 network to detect objects

    Args:
        input_image: absolute path for the image for detection
        output_image: absolute path for the saving of resulted image
        threshold: the minimum confidence that detects an object
    '''
    
    net = YOLOv2_net(is_training=False, trainable=False)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
        gpu_options=gpu_options))

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    checkpoint_dir = cfg.TRAIN.TRAINED_DIR
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Sucessfully restore the pretrained model!')
    else:
        print('Could not restore the pretrained_model!')

    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    org_h, org_w = image.shape[:2]

    sized_image= resize_image_keep_ratio(image/255.0, NET_WIDTH, NET_HEIGHT)
    
    sized_image = sized_image[np.newaxis, :]

    outputs = net.get_output('conv30')

    predicts = sess.run(outputs, feed_dict={net.data: sized_image})

   
    # We only use batch_size = 1
    predicts = np.squeeze(predicts, axis=0) 

    # Flatten the data
    predicts = np.reshape(predicts, (-1))

    # The size of output feature map
    output_w = NET_WIDTH / 32
    output_h = NET_HEIGHT / 32
    box_num = cfg.TRAIN.BOX_NUM
    class_num = len(cfg.TRAIN.CLASSES)
    box_info_len = 5 + class_num

    # Reshape predicts to 2 dimensions, each row for a box
    # The boxes w.r.t a specified grid in feature map are adjacent in y-axis
    boxes = np.reshape(predicts, (output_w * output_h * box_num, box_info_len))

    # Restore the boxes from (tx, ty, tw, th) -> (xc, yc, w, h)
    boxes = restore_boxes(boxes, output_w, output_h)

    # Correct the boxes to fit the origin image size
    boxes = correct_boxes(boxes, org_w, org_h, NET_WIDTH, NET_HEIGHT)
    
    # Retain those boxes with enough confidence
    BB = filter_boxes(boxes, threshold)

    # Apply nms
    nms_BB = nms(BB, 0.3)

    result_image = draw_results(nms_BB, image, NET_WIDTH, NET_HEIGHT)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_image, result_image)

if __name__ == '__main__':
    input_image = "web3.jpg"
    output_image = "test.jpg"

    detect(input_image, output_image, 0.24)

