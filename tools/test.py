from inference import detect
from config.config import cfg
from datasets.factory import get_imdb
import numpy as np
import os

#DATASET = 'kitti'
#DATASET = 'pascal_voc'

def test_net(dataset, model):
    '''Test the performance of your model

    Args:
        dataset: the name for dataset
        model: the absolute path for trained model

    Returns:
        all_boxes[cls][image_name]: [xc, yc, w, h, confidence]
    '''

    name = dataset + '_test'
    imdb = get_imdb(name)
    image_names = imdb.image_names
    all_boxes = [[[ ] for _ in xrange(num_images)]
            for _ in xrange(imdb.num_classes)]

    for im_index in image_names:
        BB = detect(im_index, None, model, False, 0.24)
        for b in BB:
            cls_index = b['class']
            bbox = b['box']
            confidence = b['confidence']
            info_list = [bbox[0], bbox[1], bbox[2], bbox[3], confidence]
            if all_boxes[cls_index][image_names] == []:
                all_boxes[cls_index][image_names] = np.hstack((
                        all_boxes[cls_index][image_names], info_list))
            else:
                all_boxes[cls_index][image_names] = np.vstack((
                        all_boxes[cls_index][image_names], info_list))

    return all_boxes

def parse_args():
    '''Parse input arguments

    '''
    
    parser = argparse.ArgumentParser(description = 'Test the model')
    parser.add_argument('--dataset', dest='dataset',
            help='the dataset name', default=None, type=str)

if __name == '__main__':
    args = parse_args()
    dataset = args.dataset
    model = os.path.join(cfg.TRAIN.TRAINED_DIR, dataset)

    test_net(dataset, model)
