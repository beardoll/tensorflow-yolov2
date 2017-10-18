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
* 以带BN的卷积层为例，说明参数的存储，可参考[parser.c](https://github.com/pjreddie/darknet/blob/cd5d393b46b59dc72a5150436e70fa91a2918b2d/src/parser.c).从函数`save_convolutional_weights`可以看到，模型的写入顺序是先`biases`->`scales`->`rolling_mean`->`rolling_variance`->`weights`。因此，按相同的顺序，计算好各个参数的个数，就可以顺利读取了。

部分模块说明：
* class `graph`: 根据darknet的cfg文件，将模型结构存储，主要是保存每一层的输入输出大小。如果当前层的输入和前面某些层相关（Resnet的dropout结构），那么通过这个类就可以很方便地得到输入数据的维度。
* function `convert_tf`: `count`变量用于记录当前读取的数据段（`net_weights`）的起始位置（注意`net_weights`已经去掉了前面的4个参数）。`count`每次需要移动多少主要是根据cfg文件提供的模型参数来定。
命名规则上，我只是简单地为每一个层都单独分配一个号码。比如前三层结构是input-conv-conv，那么这三层的名字将分别是'net1', 'conv2', 'conv3'.

备注：目前`convert`函数只支持部分层的参数读取，如果需要增加其它的层，可以参考darknet里的`parser.c`.

### 数据读取(`DataProducer.py`)
* 我们默认数据根据其用途存放，比如训练数据就存放在`train`文件夹下。因此类的初始化参数`image_set`不仅标明了数据的用途，还标明了其存放位置。另外，我们只对训练数据做图像增强。
* function `read_annotations`: 由于我之前用darknet做训练时，按照其要求先将数据转化成[cls_idx, xc, yc, w, h]格式，这里cls_idx是目标所属的类对应的标号，xc, yc是bounding box的中心，w和h是
候选框的长宽，它们对以宽/高进行了归一化。所以在这个函数中我设置了两种读取annotation的方式，刚刚说的就是`screen=True`；而`screen=False`，则需要重新读取原始annotation文件。
* 图像增强：
    * 先对图像的长宽比，大小进行扰动（`new_ratio`, `scale`），请注意这里`scale`是针对输出图像而言的，也就是根据训练所需的图像大小而言的。
    * 将扰动后的图像放入输出图像容器中，放置的位置是随机的，由`dx`和`dy`决定。下面我画一个图来说明，这里假设resize之后的图像比输出图像要大，因此我们只能裁出其中一部分来作为输出（dx，dy都是负数）：
<div align=center><img width="600" height="400" src="intro_material/image_processing.png"/></div>
    * 然后是在HSV空间随机扰动hue, saturation以及exposure。那部分代码写得比较长，其实就是RGB->HSV->RGB的一个过程。
    * 最后，根据对图像的扰动，对相应的gt_box也需要做相应的扰动。`box_x_scale`和`box_y_scale`是将`sized image`的量度转化到`processed image`中去，因为标签数据是以原始图像为基准进行归一化的，如上所述，
现在是将resize之后的图像塞到输出图像中，因此bounding box的归一化也必须针对输出图像（processed image）。由此也可以推知`box_x_delta`和`box_y_delta`是如何计算的。

### Reorg层（也叫passthrough层）
这一层有点类似于GoogLeNet的inception，就是将来自不同分支的feature map按照channel轴拼接到一起，要求这些分支的feature map大小是一样的。Darknet的想法是把浅层特征和深层特征结合到一起，提高网络的表达能力。
其中深层特征的feature map大小只有浅层特征的一半，因此需要将浅层特征的大小变为一半（长和宽, stride=2），思想是把相邻的元素分开放在不同的channel。我们用如下一张图来说明：

<div align=center><img width="600" height="400" src="intro_material/reorg.png"/></div>

具体的操作可以查看`reorg_op.cc`的`shard`部分。`shard`是个多线程管理模块，你可以不必知道它是如何运作的，只需要知道如何读取feature map某个位置的元素即可。在那里代码看起来很复杂，那是因为我
是将darknet关于reorg层的实现[reorg_cpu](https://github.com/pjreddie/darknet/blob/master/src/blas.c)给照搬过来，前面已经说过darknet和tensorflow的blob数据格式是不一样的。对照着darknet的代码
来看会比较好理解。

从图中也可以看到，这样划分的方式似乎还是不太好。比如1，2，5，6应该分别放在四个不同的channel，但是darknet上实现的就是如图所示那样。

### Region层
这一层的功能主要是为了计算loss。总结一下大概是做了两件事情（region_op.cc）：
* 对于每个predicted box，寻找与其最匹配的gt_box，并且计算overlap。若overlap > thresh，则暂时将confidence loss设为0；否则计算confidence loss = predicted_confidence - 0
(因为这些box的confidence应该是0，也就是没有检测到任何目标)。
* 对于每个gt_box，寻找与其overlap最大的predicted box（唯一），并且为该predicted box计算confidence loss, regression loss, classification loss等。也就是说，只有部分的predicted box
会得到训练，而其余的摇摆人（虽然overlap > thresh，但是不跟gt_box绑定），则不计算loss（也就是不对它们进行训练）。可以看出这是一种“精英训练”的策略。

接下来说明一下loss是如何计算。我们首先关注一下最后的卷积层，`conv30`。这一层其实已经给出了预测结果。它有box_num * (class_num + 5)那么多个channel，其中`5`包含了：目标所属的类，目标的
bounding box，每个box都有5+class_num那么多预测值。对于feature map中的每个像素点（对应于原图中的某一块区域，感受野），它都会产生box_num个box信息。参考论文可知，我们会预先对训练数据的bounding
box进行聚类，得到box_num种bounding box的大小作为先验信息（长，宽）。实际上，预测的bounding box的信息是针对于先验信息的偏移。可以用如下的图来解释：
<div align=center><img width="600" height="400" src="intro_material/region.png"/></div>
