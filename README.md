# YOLOv2的tensorflow版本

原文请参考：[YOLOv2 paper](https://pjreddie.com/media/files/papers/YOLO9000.pdf)。

注：本工程只实现了YOLOv2，并没有实现YOLO9000。

## 安装

```Shell
cd $YOLO_ROOT/lib
./make.sh
```
以上步骤是为了生成region layer和reorg layer的动态链接库（.so），供tensorflow调用。

## 代码说明

### `tools/convert.py`

作用：将darknet格式的模型文件转化为python格式（.npy）。

darknet格式
* 以float型数组格式存储，其blob数据格式与Caffe一样，即卷积层参数为[c_o, c_i, h, w]，全连接层参数为[c_o, c_i]。
* 模型参数从数组的第5个数据开始，前四个参数分别是：major, minor, revision and net.seen，可以忽略这四个参数的作用。
* 以带BN的卷积层为例，说明参数的存储，可参考[parse.c](https://github.com/pjreddie/darknet/blob/cd5d393b46b59dc72a5150436e70fa91a2918b2d/src/parser.c).从函数`save_convolutional_weights`可以看到，模型的写入顺序是先`biases`->`scales`->`rolling_mean`->`rolling_variance`->`weights`。因此，按相同的顺序，计算好各个参数的个数，就可以顺利读取了。

部分模块说明：
* class `graph`: 根据darknet的cfg文件，将模型结构存储，主要是保存每一层的输入输出大小。如果当前层的输入和前面某些层相关（Resnet的dropout结构），那么通过这个类就可以很方便地得到输入数据的维度。
* function `convert_tf`: `count`变量用于记录当前读取的数据段（`net_weights`）的起始位置（注意`net_weights`已经去掉了前面的4个参数）。`count`每次需要移动多少主要是根据cfg文件提供的模型参数来定。
命名规则上，我只是简单地为每一个层都单独分配一个号码。比如前三层结构是input-conv-conv，那么这三层的名字将分别是'net1', 'conv2', 'conv3'.