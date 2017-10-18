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