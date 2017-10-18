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
* 以带BN的卷积层为例，说明参数的存储，可参考[parse.c](https://github.com/pjreddie/darknet/blob/cd5d393b46b59dc72a5150436e70fa91a2918b2d/src/parser.c).