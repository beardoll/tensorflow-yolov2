import numpy as np
import tensorflow as tf
import reorg_layer.reorg_op as reorg_op
import reorg_layer.reorg_op_grad
import region_layer.region_op as region_op
import region_layer.region_op_grad

from config.config import cfg

DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True, is_training=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.is_training = is_training
        self.pretrained_variable_list = []
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        self.pretrained_variable_list.append(var)
                        print "assign pretrain model "+subkey+ " to "+key
                    except ValueError:
                        print "ignore "+key
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def _variable_with_weight_decay(self, name, shape, initializer, wd, trainable):
        '''Initialize variables with weight decay

        Args:
            name: name of the variable
            shape: shape of variable ,list of ints
            initializer: initializer for varible (always truncated norm dist)    
            wd: add L2-loss decay multiplied by this float. If None, weight
            decay is not added for this variable
            trainable: whether this variable can be trained
        
        Returns:
            Variable Tensor
        '''

        var = tf.get_variable(name, shape, initializer=initializer, trainable =
                trainable)

        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                name='weight_loss')
        tf.add_to_collection('weight_decay', weight_decay)

        return var


    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name,
            activation='leaky_relu', padding=DEFAULT_PADDING, group=1, 
            trainable=True, batchnorm=False):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            
            if cfg.TRAIN.has_key('DECAY'):
                # Using weight_decay
                kernel = self._variable_with_weight_decay('weights', [k_h, k_w,
                    c_i/group, c_o], init_weights, cfg.TRAIN.DECAY, trainable)
            else:
                kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            
            if batchnorm:
                # Apply batch normalization
                conv = self.batch_normalization(conv,
                        is_training=self.is_training, name = name, is_conv_out=True,
                        decay=0.999)
            else:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = tf.nn.bias_add(conv, biases)

            if activation == 'leaky_relu':
                return self.leaky_relu(conv)
            elif activation == 'relu':
                return tf.nn.relu(conv, name=scope.name)
            elif activation == 'softmax':
                return tf.nn.softmax(conv, name=scope.name)
            elif activation == 'linear':
                return conv
            else:
                raise ValueError('Use undefined activation function!')


    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)


    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)

            if cfg.TRAIN.has_key('DECAY'):
                # Using weight decay
                weights = self._variable_with_weight_decay('weights', [dim,
                    num_out], init_weights, cfg.TRAIN.DECAY, trainable)

            else:
                weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            
            init_biases = tf.constant_initializer(0.0)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they are not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW)
            if input_shape[1] == 1 and input_shape[2] == 1:
                # feature map is (b, 1, 1, c)
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def reorg(self, input, stride, name):
        return reorg_op.reorg(input, stride, name=name)

    @layer
    def region(self, input, labels, seen, name):
        return region_op.region(input, labels, cfg.TRAIN.ANCHORS, seen, cfg.TRAIN.NOOBJECT_SCALE, 
                cfg.TRAIN.OBJECT_SCALE, cfg.TRAIN.COORD_SCALE, cfg.TRAIN.CLASS_SCALE,
                len(cfg.TRAIN.CLASSES), cfg.TRAIN.BOX_NUM, cfg.TRAIN.THRESH, name=name)

    def batch_normalization(self, input, is_training, name, 
            is_conv_out=True, decay = 0.99):
        '''Implementing batch normalization for training and testing
    
        Args:
            is_training: training process or testing process
            is_conv_out: whether to apply BN in convolutional layer
            decay: for pop_mean and pop_var
        '''

        shape = input.get_shape()[-1]

        ones_initializer = tf.constant_initializer(1.0)
        zeros_initializer = tf.constant_initializer(0.0)

        scale = self.make_var('scale', [shape], ones_initializer, True)
        offset = self.make_var('offset', [shape], zeros_initializer, True)
        pop_mean = self.make_var('mean', [shape], zeros_initializer, False)
        pop_var = self.make_var('variance', [shape], ones_initializer, False)

        if is_training:
            if is_conv_out:
                batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2])
            else:
                batch_mean, batch_var = tf.nn.moments(input, [0])

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean *
                    (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 -
                decay))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input, batch_mean, batch_var,
                        offset, scale, 0.001, name=name)
        else:
            return tf.nn.batch_normalization(input, pop_mean, pop_var, offset,
                    scale, 0.001, name=name)

    def leaky_relu(self, input, alpha=0.1):
        '''leaky relu

        if x > 0:
            return x
        else:
            return alpha * x
        Args:
            input: Tensor
            alpha: float
        Returns:
            y: Tensor
        '''
        x = tf.cast(input, dtype=tf.float32)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=tf.float32)
        return 1.0 * mask * x + alpha * (1.0 - mask) * x

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
