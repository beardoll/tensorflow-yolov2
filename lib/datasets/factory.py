from dataset.kitti import kitti
from dataset.pascal_voc import pascal_voc

"""
    You can set up your own dataset here
"""

__sets = {}

# set up kitti datasets
for split in ['train', 'test']:
    name = 'kitti_{}'.format(split)
    __sets[name] = kitti(split, data_argument = (split == 'train'))

# set up pascal_voc datasets
for split in ['train', 'test']:
    name = 'pascal_voc_{}'.format(split)
    __sets[name] = pascal_voc(split, data_argument = (split == 'train'))

def get_imdb(name):
    """Get an imdb by name"""
    if not __sets.has_key(name):
        raise KeyError("Unknown datasets: {}".format(name))
    return __sets[name]


