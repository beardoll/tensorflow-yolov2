import tensorflow as tf
import numpy as np
import region_op
import region_op_grad
import pdb


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant_initializer(0.0)
    return tf.get_variable(name, shape, initializer=initial, trainable=True)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def forward_test():
    '''Test the forward pass of region layer

    Here we set predicts as random values but  
    specify the first predict as (tx=0, ty=0, tw=0, th=0), in the box (left=0, top=0)
    And we set anchors as [prior_w=0.2, prior_h=0.2]
    Then the specify predicted box is (cx=0.25, cy=0.25, w=0.1, h=0.1)
    
    The label is set to (cx=0.25, cy=0.25, w=0.1, h=0.1)
    That means the label is only active in the box (left=0, top=0)

    If the forward pass is ok, there is exactly one predicted box has 1.0 iou with the
    label, and the class loss is -0.3
    For other predicted boxes, the iou and other losses are randomized
    However, you can track the computation of those boxes, because those boxes
    will never be responsible for the label box, if you diable the "seen"
    (means set seen larger than 12800), then those boxes have only the obj loss
    If you enable "seen", those boxes will have coords loss.
    
    Any way, if you find out that the final loss is as you expected, then the
    forward pass is ok. From now on we believe that the forward pass is ok
    acoording to our test.
    '''

    predict_array = np.random.rand(1, 2, 2, 7)
    predict_array[0, 0, 0, :] = np.array([0, 0, 0, 0, 0.7, 0.5, 0.5])
    label_array = np.zeros((1, 30, 5), dtype = np.float32)
    label_array[0, 0,: ] = np.array([0, 0.25, 0.25, 0.1, 0.1])
    anchor_array = np.array([0.2, 0.2])
    
    noobject_scale = 1
    object_scale = 1
    class_scale = 1
    coord_scale = 1

    # set seen > 12800 if you want to disable it
    seen = 0

    thresh = 0.6

    class_num = 2
    box_num = 1

    predict = tf.convert_to_tensor(predict_array, dtype = tf.float32)
    label = tf.convert_to_tensor(label_array, dtype = tf.float32)
    anchor = tf.convert_to_tensor(anchor_array, dtype = tf.float32)

    output = region_op.region(predict, label, anchor, noobject_scale,
            object_scale, coord_scale, class_scale, class_num, box_num, seen, thresh)

    init = tf.initialize_all_variables()
            
    with tf.Session() as sess:
        sess.run(init)
        sess.run(output)



def backward_test():
    '''Test the backward pass of region layer

    We construct one convolutional layer with linear activation function,
    and feed the activated values to the region layer to get predicts.

    We set four labels centered in different grids in feature map, that means
    we expect in each grid, we have exactly one predicted box to be close to
    the corresponding label.

    We generate 3 boxes in each grid, and the class number is 2, then each box
    has information with length (2 + 5)

    We expected the average iou as 1, the class as 1, the confidence(obj) as 1
    and the average obj(No obj) is 1/3, for only 1 box in 3 boxes should be
    responsible for the label. We also expect the recall as 1.0

    However, we cannot closly reach the targets, may be some problems still
    exist in the forward-backward pass.
    '''

    image = np.random.rand(1, 2, 2, 3)
    label = np.zeros((1, 30, 5), dtype = np.float32)
    label[0, 0, :] = np.array([1, 0.25, 0.25, 0.5, 0.5])
    label[0, 1, :] = np.array([0, 0.75, 0.25, 0.5, 0.5])
    label[0, 2, :] = np.array([1, 0.25, 0.75, 0.5, 0.5])
    label[0, 3, :] = np.array([0, 0.75, 0.75, 0.5, 0.5])
    
    data = tf.convert_to_tensor(image, dtype = tf.float32)
    target = tf.convert_to_tensor(label, dtype = tf.float32)

    W = weight_variable([1, 1, 3, 21])
    b = bias_variable([21], "bias1")

    box_num = 3

    class_num = 2

    noobject_scale = 1
    object_scale = 5
    class_scale = 1
    coord_scale = 1
    seen = 30000
    thresh = 0.6

    anchor = [2, 0.3, 1, 0.5, 0.6, 0.3]

    prior_size = tf.convert_to_tensor(anchor, dtype = tf.float32)
    
    output1 = conv2d(data, W)

    output1 = tf.nn.bias_add(output1, b, name=None)
   
    output3 = region_op.region(output1, target, prior_size, noobject_scale,
            object_scale, coord_scale, class_scale, class_num, box_num, seen, thresh)

    #output2 = tf.reshape(output2, [1, -1])

    loss = tf.reduce_mean(tf.square(output3))

    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for step in xrange(100000):
            #out = sess.run([output2])
            loss1, _ = sess.run([loss, train])
            #print(loss1)



if __name__ == '__main__':
    backward_test()
    #forward_test()



