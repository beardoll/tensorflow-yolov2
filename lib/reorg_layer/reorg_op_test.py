import tensorflow as tf
import numpy as np
import reorg_op
import reorg_op_grad

def weight_variable(shape):
    '''Weights initializer

    '''

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''Bias initializer

    '''

    initial = tf.constant_initializer(0.0)
    return tf.get_variable("bias", shape, initializer=initial, trainable=True)


def conv2d(x, W):
    '''Convolutional op

    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def forward_test():
    '''Test the forward pass of reorg layer

    '''
    a = np.zeros((1, 4, 4, 4), dtype = np.float32)

    for i in range(64):
        # data format (b, h, w, c)
        n = i
        c = n % 4
        n = n / 4
        w = n % 4
        n = n / 4
        h = n % 4
        n = n / 4
        b = n
        a[b, h, w, c] = i;

        sess = tf.Session()
        b = sess.run(reorg(a, 2))

        a = np.transpose(a, (0, 3, 1, 2))
        b = np.transpose(b, (0, 3, 1, 2))

    print a
    print b


def backward_test():
    '''Test the backward pass of reorg layer

    We construct a simple convolutional layer as input to reorg layer
    And force the output of reorg layer to learn a ones-like mat
    '''

    # The input picture of shape (4, 4, 3), only one picture
    array = np.random.rand(1, 4, 4, 3)
    data = tf.convert_to_tensor(array, dtype = tf.float32)
    
    # kernel size: 2x2, output channels: 4
    W = weight_variable([2, 2, 3, 4])
    b = bias_variable([4])
    
    # Output of convolutional layer has shape: [1, 4, 4, 4]
    h = conv2d(data, W)
    h = tf.nn.bias_add(h, b, name=None)
    
    # Output of reorg layer has shape: [1, 2, 2, 16]
    y = reorg_op.reorg(h, 2)
    
    # The labels 
    y_data = tf.convert_to_tensor(np.ones((1, 2, 2, 16)), dtype=tf.float32)
    
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)
    
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        for step in xrange(100000):
            loss1, _ = sess.run([loss, train])
            print(loss1)
    
        print(sess.run(y))

if __name__ == '__main__':
    #forward_test()
    backward_test()
