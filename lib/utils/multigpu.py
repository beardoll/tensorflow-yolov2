import tensorflow as tf

def average_gradients(tower_grads):
    '''Average gradients for variables in tower_grads

    Args:
        tower_grads: [((grad0_gpu0, var0_gpu0), ..., (gradN_gpu0, 
        varN_gpu0)), ((grad0_gpu1, var0_gpu1), ...,  (gradN_gpu1,
        varN_gpu1)), ... ]

    Returns:
        average_grads: [(grad0, var0), ..., (gradN, varN)]
    '''

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Transform tower_grads to the format:
        # [((grad0_gpu0, var0_gpu0), ..., (grad0_gpuM, var0_gpuM), ...,
        # (gradN_gpu0, varN_gpu0), ..., (gradN_gpuM, varN_gpuM))]
        grads = [g for g, _ in grad_and_vars]

        # Average over the 'tower' dimension
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are 
        # shared across towers. So .. we will just return the first towers'
        # pointer to the Variable
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


