import tensorflow as tf

'''Test the while_loop api in tensorflow

We construct 3 loops, and try to get the times of loop
'''


def body(loop1, loop2, loop3, stop1, stop2, stop3, i):
    return loop1, loop2, loop3+1, stop1, stop2, stop3, i+1

def cond1(loop1, loop2, loop3, stop1, stop2, stop3, i):
    return loop1 < stop1


def cond2(loop1, loop2, loop3, stop1, stop2, stop3, i):
    return loop2 < stop2

def cond3(loop1, loop2, loop3, stop1, stop2, stop3, i):
    return loop3 < stop3

def body1(loop1, loop2, loop3, stop1, stop2, stop3, i):
    loop3 = 0
    l1, l2, l3, s1, s2, s3, i =  tf.while_loop(cond3, body, [loop1, loop2, loop3, stop1, stop2, stop3, i])
    l2 += 1
    return l1, l2, l3, s1, s2, s3, i


def body2(loop1, loop2, loop3, stop1, stop2, stop3, i):
    loop2 = 0
    l1, l2, l3, s1, s2, s3, i = tf.while_loop(cond2, body1, [loop1, loop2, loop3, stop1, stop2, stop3, i])
    l1 += 1
    return l1, l2, l3, s1, s2, s3, i


def test():

    loop1 = tf.constant(0, tf.int32)
    loop2 = tf.constant(0, tf.int32)
    loop3 = tf.constant(0, tf.int32)

    result = tf.constant(0, tf.int32)

    tuple_results = tf.while_loop(cond1, body2, [loop1, loop2, loop3, 3, 3, 3,
        result])

    res = tuple_results[-1]

    sess = tf.Session()
    
    print sess.run(res)

if __name__ == '__main__':
    test()
