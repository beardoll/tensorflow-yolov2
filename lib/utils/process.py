import _init_paths
import tensorflow as tf
import numpy as np
from net.yolov2_net import YOLOv2_net
from config.config import cfg
import cv2

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
    lu = b1[0:2] * (b1[0:2] >= b2[0:2]) + \
         b2[0:2] * (b1[0:2] < b2[0:2])

    rd = b1[2:4] * (b1[2:4] <= b2[2:4]) + \
         b2[2:4] * (b1[2:4] > b2[2:4])

    intersection = rd - lu

    inter_square = intersection[0] * intersection[1]
    
    # Elimated those intersection with w or h < 0
    mask = np.array(intersection[0] > 0, np.float32) * \
           np.array(intersection[1] > 0, np.float32)

    inter_square *= mask

    # Calculate boxes square
    square1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    square2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    return inter_square / (square1 + square2 - inter_square + 1e-6)

    
def resize_image_keep_ratio(image, w, h):
    '''Resize the image while keeping the width/height ratio

    The content of image will cover the center brand of sized_image
    For those uncovered areas in the resized image, we set them to 0.5 (grey
    pixels w.r.t [0, 1])
    
    Args:
        image: the original image ndarray (normalized pixels)
        w, h: the expected size of resulting image
    Returns:
        sized_image: the resized image (normalized pixels)
    '''
    sized_image = np.ones((h, w, 3), dtype=np.float32) * 0.5
    org_h, org_w = image.shape[:2]
    nw = org_w
    nh = org_h
    if float(w/org_w) < float(h/org_h):
        nw = w
        nh = int(org_h * w / org_w)
    else:
        nh = h
        nw = int(h / org_h * org_w)

    dx = int((w - nw) * 1.0 / 2)
    dy = int((h - nh) * 1.0 / 2)
    
    sized_image[dy: h - dy, dx: w - dx, :] = cv2.resize(image, (w-dx*2,
        h-dy*2))
   
    return sized_image

def resize_label_keep_ratio(boxes, org_w, org_h, sized_w, sized_h):
    '''Resize gt-boxes from w.r.t (org_w, org_h) -> (sized_w, sized_h)
    
    Here, sized_w / sized_h == org_w / org_h, the content of image is centered
    on the resized image, and the original coords of gt-boxes are w.r.t the
    content of image, we should extend them to the whole image

    Args:
        boxes: gt-boxes, with 5 elements per row, (xc, yc, w, h)
        org_w, org_h: the original size of image
        sized_w, sized_h: the resized size
    Returns:
        the resized_boxes
    '''
    nw = 0
    nh = 0
    if float(sized_w / org_w) < float(sized_h / org_h):
        nw = sized_w
        nh = int(org_h * sized_w / org_w)
    else:
        nh = sized_h
        nw = int(org_w * sized_h / org_h)

    for box in boxes:
        box[0] = box[0] * (nw*1.0/sized_w) + (sized_w - nw) / 2.0 / sized_w
        box[1] = box[1] * (nh*1.0/sized_h) + (sized_h - nh) / 2.0 / sized_h
        box[2] /= sized_w * 1.0 / nw
        box[3] /= sized_h * 1.0 / nh

    return boxes







