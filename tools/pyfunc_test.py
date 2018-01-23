import tensorflow as tf
import numpy as np

'''Test the py_func api in tensorflow

py_func is a wrapper for embedding python op in tensorflow graph
However, py_func cannot feed back gradients...
'''

def add_two_vector(a, b):
    c = np.zeros((a.shape[0]), dtype=np.float32)
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

sess = tf.Session()
a = tf.placeholder(tf.float32, shape=(5, ))
b = tf.placeholder(tf.float32, shape=(5, ))

add_op = tf.py_func(add_two_vector, [a, b], [tf.float32])

c = sess.run(add_op, feed_dict={a: np.array([1, 1, 1, 1, 1], dtype=np.float32), \
            b: np.array([2, 2, 2, 2, 2], dtype=np.float32)})

print c
