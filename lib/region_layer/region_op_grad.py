import tensorflow as tf
from tensorflow.python.framework import ops
import region_op

@ops.RegisterShape("Region")
def _region_shape(op):
    '''Shape function for region op

    '''
    dims_data = op.inputs[0].get_shape().as_list()

    box_num = op.get_attr('box_num')

    class_num = op.get_attr('class_num')

    output_shape = tf.TensorShape([None, None, None, box_num * (class_num + 5)])
    
    return [output_shape]

@ops.RegisterGradient("Region")
def _region_grad(op, grad):
    ''' The gradients for region
    Args:
        op: The region operation that we are differentiating, which we can use to 
            find the inputs and outputs for the original op
        grad: Gradient with respect to the output of the region op
    Returns:
        Gradients with respect to the input
    '''

    data = op.inputs[0]

    box_num = op.get_attr("box_num")
    class_num = op.get_attr("class_num")

    # compute grad
    data_grad = region_op.region_grad(data, grad, class_num, box_num)

    return [data_grad, None, None]
