from ConfigParser import ConfigParser
from collections import OrderedDict
import os
import sys
import numpy as np
from config.config import cfg
import _init_paths

class uniqdict(OrderedDict):
    '''Giving each layer a unique name

    The instances in cfg files are read into sections.
    e.g: Net_1, Convolutional_2, pool_3 ....
    '''
    _unique = 0
    def __setitem__(self, key, val):
        if isinstance(val, OrderedDict):
            self._unique += 1
            key += "_" + str(self._unique)

        OrderedDict.__setitem__(self, key, val)

class layer(object):
    '''Store the information of import layers

    '''
    def __init__(self, name, height, width, channels):
        '''Initialize the layer
        Args:
            name: "net", "conv", "pool", "route", "reorg" or "fc" + index
            height, width, channels: output feature maps'
        '''

        self.__name = name
        self.__height = height
        self.__width = width
        self.__channels = channels
    
    def get_para(self):
        return [self.__name, self.__height, self.__width, self.__channels]

class Graph(object):
    '''Store the layers desgnated by .cfg file
    
    '''
    def __init__(self):
        self.__layers = []
        self.__num = 0  # the number of layers

    def add_layer(self, layer):
        '''Add the layer to the rear of __layers

        '''
        _, height, width, channels = layer.get_para()
        print "height: {}, width: {}, channels: {}".format(height, width, channels)
        self.__layers.append(layer)
        self.__num += 1

    def get_layer(self, index):
        '''Get the layer in __layers according to index

        '''
        return self.layers[index-1]

    def get_graph_depth(self):
        ''' get the the depth of graph (number of layers)
        
        '''
        return self.num

def convert_tf(cfg_file, weight_file, output_file):
    '''Convert .weight file to .npy file.

    Args:
        cfg_file, weight_file: from darknet.
        output_file: format of '.npy', using two-level dict to store the data
        
    '''

    # Parse the cfg file into instances.
    # The instance represents a unique layer.
    parser = ConfigParser(dict_type = uniqdict)
    parser.read(cfg_file)

    # Construct empty graph.
    graph = Graph()

    # Using two-level dictionary to store the params,
    # e.g. data_dict['conv1']['weights'] represents weights in conv1 layer
    data_dict = {}

    # Only part of layers have params, we give these layers unique index starting from 1
    layer_idx = 1

    # Extract flags
    # The first four elements in weight file are major, minor, revision and net.seen
    net_weights_int = np.fromfile(weight_file, dtype=np.int32) 
    trans_flag = (net_weights_int[0] > 1000 or net_weights_int[1] > 1000)
    print trans_flag

    # load weights
    net_weights_float = np.fromfile(weight_file, dtype=np.float32)
    net_weights = net_weights_float[4:]
    print net_weights.shape

    # The current position in net_weights
    count = 0

    for section in parser.sections():
        if count == net_weights.shape[0]:
            print "All weights have been loaded"
            break

        print "Loading layer: {}".format(section)

        # Extract the layer name
        _section = section.split('_')[0]          
        
        if _section in ["crop", "cost", "region"]:
            # These layers have no weights, we just split them
            continue

        # Extract all items in this layer
        items = dict(parser.items(section))
        if _section == 'net':
            l = layer('net', int(items['height']), int(items['width']), int(items['channels']))
            graph.add_layer(l)

        elif _section == 'convolutional':
            conv_filters = int(items['filters'])
            conv_size = int(items['size'])

            layer_num = graph.get_graph_depth()
            
            assert layer_num != 0, 'WTF?? Why there are no layers!!'

            # Get the input layer for current convolutional layer.
            # Commonly the size of feature map stays unchanged after
            # convolution.
            last_layer = graph.get_layer(layer_num)
            _, h_in, w_in , c_in = last_layer.get_para()
            conv_width = w_in
            conv_height = h_in

            # Mark whether BN is applied in current layer
            batchnorm_followed = False
            if 'batch_normalize' in items and items['batch_normalize']:
                batchnorm_followed = True

            op_name = 'conv' + str(layer_idx)
            layer_idx += 1
            data_dict[op_name] = {}

            # If with BN, the params are stored in order:
            #     biases -> scale -> mean -> variance -> weights
            # Without BN, the order is: biases -> weights

            bias_size = conv_filters
            conv_bias = np.reshape(net_weights[count: count+bias_size], (bias_size, ))
            count += bias_size

            if batchnorm_followed == True:
                # Rename 'bias' to 'offset' for it's applied in BN layer
                data_dict[op_name]['offset'] = conv_bias

                bn_scales = np.reshape(net_weights[count: count+bias_size], (bias_size, ))
                data_dict[op_name]['scale'] = bn_scales
                count += bias_size

                bn_rolling_mean = np.reshape(net_weights[count: count+bias_size], (bias_size, ))
                data_dict[op_name]['mean'] = bn_rolling_mean
                count += bias_size

                bn_rolling_variance = np.reshape(net_weights[count: count+bias_size], (bias_size, ))
                data_dict[op_name]['variance'] = bn_rolling_variance
                count += bias_size
            else:
                # No BN layer, just load bias
                data_dict[op_name]['biases'] = conv_biases
            
            # Load weights
            # Original weights format: [c_o, c_i, h, w]
            # Different from tensorflow format: [h, w, c_i, c_o]
            dims = (conv_filters, c_in, conv_size, conv_size)
            weights_size = np.prod(dims)
            conv_weights = np.reshape(net_weights[count: count+weights_size], dims)
            data_dict[op_name]['weights'] = np.transpose(conv_weights, (2, 3, 1, 0))
            count += weights_size
            
            new_layer = layer('conv', conv_height, conv_width, conv_filters)
            graph.add_layer(new_layer)

        elif _section == 'connected':
            '''Fc layer

            '''
            op_name = 'fc' + str(layer_idx)
            data_dict[op_name] = {}
            layer_idx += 1

            layer_num = graph.get_graph_depth()

            assert layer_num != 0, 'Layer num should not be 0!'

            last_layer = graph.get_layer(layer_num)
            _, h_in, w_in, c_in = last_layer.get_para()
            
            # The input size is the total pixels of input feature maps
            input_size = h_in * w_in * c_in

            # The output size has been desgnated in .cfg file
            output_size = int(items['output'])

            # The params order: biases -> weights
            fc_bias = np.reshape(net_weights[count: count + output_size], (output_size, ))
            data_dict[op_name]['biases'] = fc_bias
            count += output_size

            # Original data format: (c_o, c_i)
            # Tensorflow data format: (c_i, c_o)
            dims = (output_size, input_size)
            
            fc_weights = None
            if tranFlag:
                fc_weights = np.reshape(net_weights[count: count + input_size * output_size], (dims[1], dims[0]))
            else:
                fc_weights = np.reshape(net_weights[count: count + input_size * output_size], dims)

            # (c_o, c_i) -> (c_i, c_o)
            data_dict[op_name]['weights'] = np.transpose(fc_weights, (1, 0))
            count += input_size * output_size    
            
            new_layer = layer('fc', 1, 1, output_size)
            graph.add_layer(new_layer)

        elif _section == 'maxpool' or _section == 'avgpool':
            '''Pooling layer

            By default, size == stride, and the output size of feature map 
            is divided by stride.
            If not, please modify the relative computation about shape.
            '''

            layer_num = graph.get_graph_depth()
            assert layer_num != 0, 'Layer number should not be zero!'
            last_layer = graph.get_layer(layer_num)

            _, h_in, w_in, c_in = last_layer.get_para()
            
            stride = int(items['stride'])

            new_layer = layer('pool', h_in/stride, w_in/stride, c_in)
            graph.add_layer(new_layer)

            layer_idx += 1

        elif _section == 'route':
            '''Route layer

            Similar to 'concate' in inception model.
            '''

            layers = items['layers'].split(',')
            layer_num = graph.get_graph_depth()
            assert layer_num != 0, 'Layer num should not be zero'

            # To apply route, all maps should have the same width and height
            # And the output channels are the sumation of the channels of them

            channels = 0
            width = 0
            height = 0
            for offset in layers:
                offset = int(offset)
                current_layer = graph.get_layer(layer_num + offset)
                _, h, w, c = current_layer.get_para()
                if(width != 0):
                    # Be sure that all routed layer has the same width
                    assert width == w, "Route layer should guarantee the unchanged shape of feature map"
                else:
                    width = w
                if(height != 0):
                    # Be sure that all routed layer has the same height
                    assert height == h, "Route layer should guarantee the unchanged shape of feature map"
                else:
                    height = h

                channels += c
            
            new_layer = layer('route', height, width, channels)
            graph.add_layer(new_layer)
            layer_idx += 1

        elif _section == 'reorg':
            '''Reorg layer

            To re-organize the shape of feature maps (expand)
            '''

            layer_num = graph.get_graph_depth()
            assert layer_num != 0, 'Layer num should not be zero'
            last_layer = graph.get_layer(layer_num)

            _, h_in, w_in, c_in = last_layer.get_para()

            stride = int(items['stride'])

            new_layer = layer('reorg', h_in/stride, w_in/stride, c_in*stride*stride)

    print count
    assert count == net_weights.shape[0], 'There are still some paras not be loaded!!'

    np.save(output_file, data_dict)
        

if __name__ == '__main__':
    cfg_file = os.path.join(cfg.PRETRAINED_PATH, 'darknet', 'yolo-voc.cfg')
    weights_file = os.path.join(cfg.PRETRAINED_PATH, 'darknet',
            'darknet19_448.conv.23')
    output_file = os.path.join(cfg.PRETRIANED_PATH, 'npy', 'yolov2.npy')
    
    convert_tf(cfg_file, weights_file, output_file)
