import tensorflow as tf
from tensorflow.python.framework import ops
import reorg_op
import pdb

@ops.RegisterShape("Reorg")
def _reorg_shape(op):
    '''
    Shape function for reorg op
    '''
    dims_data = op.inputs[0].get_shape().as_list()

    channels = dims_data[3]
    batches = dims_data[0]
    height = dims_data[1]
    width = dims_data[2]

    stride = op.get_attr('stride')
    output_shape = tf.TensorShape([batches, height/stride, width/stride, channels*stride*stride])
    return [output_shape]

@ops.RegisterGradient("Reorg")
def _reorg_grad(op, grad):
    ''' The gradients for reorg
    Args:
        op: The reorg operation that we are differentiating, which wen can use to 
            find the inputs and outputs for the original op
        grad: Gradient with respect to the output of the reorg op
    Returns:
        Gradients with respect to the input
    '''
    
    stride = op.get_attr('stride')

    # compute grad
    data_grad = reorg_op.reorg_grad(grad, stride)

    return [data_grad]    # List of one tensor, since we have one input
