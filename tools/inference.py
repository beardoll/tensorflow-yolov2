import _init_paths
import tensorflow as tf
import numpy as np
from net.yolov2_net import YOLOv2_net
from config.config import cfg
import cv2

# The network size, according to the input size of final hundreds of training
NET_WIDTH = 416
NET_HEIGHT = 416

def softmax(logits):
    '''Calculate softmax(logits)
    
    Args:
        logits: np.array
    Returns:
        probs: np.array, probs for each class
    '''

    num = logits.shape[0]

    max_value = np.max(logits)

    probs = np.exp(logits - max_value)
    probs /= np.sum(probs)

    return probs

def logistic(x):
    return 1.0 /(1.0 + np.exp(-x))


def iou(box1, box2):
    '''Calculate iou between box1 and box2

    Args:
        box1, box2: np.array, (xc, yc, w, h)
    Returns:
        iou: iou value
    '''

    # Transform (xc, yc, w, h) -> (x1, y1, x2, y2)
    b1 = np.array([box1[0] - box1[2] / 2, 
                   box1[1] - box1[3] / 2,
                   box1[0] + box1[2] / 2,
                   box1[1] + box1[3] / 2], dtype=np.float32)

    b2 = np.array([box2[0] - box2[2] / 2, 
                   box2[1] - box2[3] / 2,
                   box2[0] + box2[2] / 2,
                   box2[1] + box2[3] / 2], dtype=np.float32)

    # Calculate the left-up points and right-down points 
    # of overlap areas
    lu = box1[0:2] * (box1[0:2] >= box2[0:2]) + \
         box2[0:2] * (box1[0:2] < box2[0:2])

    rd = box1[2:4] * (box1[2:4] <= box2[2:4]) + \
         box2[2:4] * (box1[2:4] > box2[2:4])

    intersection = rd - lu

    inter_square = intersection[0] * intersection[1]
    
    # Elimated those intersection with w or h < 0
    mask = np.array(intersection[0] > 0, np.float32) * \
           np.array(intersection[1] > 0, np.float32)

    inter_square *= mask

    # Calculate boxes square
    square1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    square2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_square / (square1 + square2 - inter_square + 1e-6)


def filter_boxes(boxes, threshold, map_w, map_h):
    '''Get delegate box in boxes

    confidence = max(prob) * Ptr(obj)
    Only those confidence > threshold will be recorded

    Args:
        boxes: coords(4) + obj(1) + class_num
        map_w, map_h: the width and height of the feature map, they are used
            for restore the coords in boxes, for each box is w.r.t specified
            grid in feature map 
    
    Returns:
        BB: dict, with 'box', 'confidence', 'class'
    '''
    
    box_num = cfg.TRAIN.BOX_NUM

    BB = []
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

        obj = logistic(box[4])

        logits = box[5:]
        probs = softmax(logits)
        max_prob = np.max(probs)

        confidence = obj * max_prob
        if confidence > threshold:
            temp = {}
            box_x = (logistic(box[0]) + w) / map_w
            box_y = (logistic(box[1]) + h) / map_h
            box_w = (np.exp(box[2]) * prior_w) / map_w
            box_h = (np.exp(box[3]) * prior_h) / map_h

            print box_x, box_y, box_w, box_h

            temp['box'] = np.array([box_x, box_y, box_w, box_h])
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

        print rescale_bbox

        print int(x1), int(y1), int(x2), int(y2)

        org_img = cv2.rectangle(org_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255,
            0), 2)
        #cv2.putText(org_img, class_name + " " + str(confidence), (int(x1),
        #    int(y1)-10), font, 0.75, (0, 255, 0), 1)

    return org_img


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
    
    sized_image = np.array(image, dtype=np.float32)
    sized_image /= 255.0
    sized_image = cv2.resize(sized_image, (NET_WIDTH, NET_HEIGHT))
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

    # Retain those boxes with enough confidence
    BB = filter_boxes(boxes, threshold, output_w, output_h)

    # Apply nms
    nms_BB = nms(BB, 0.3)

    result_image = draw_results(nms_BB, image, NET_WIDTH, NET_HEIGHT)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_image, result_image)

if __name__ == '__main__':
    input_image = "web3.jpg"
    output_image = "test.jpg"

    detect(input_image, output_image, 0.24)

