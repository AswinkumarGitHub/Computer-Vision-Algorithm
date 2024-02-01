# SSD

> [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

<!-- [ALGORITHM] -->

## Abstract

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300×300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500×500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143998553-4e12f681-6025-46b4-8410-9e2e1e53a8ec.png"/>
</div>


## Training Data: COCO

## Training Techniques:
        - SGD with Momentum
        - Weight Decay

## Training Resources: 8x V100 GPUs
      
## Architecture:
        - VGG

## Results and models of SSD

| Backbone | Size | Style | Lr schd | Mem (GB) | Inf time (fps) | box AP |           Config           |                                                                                                             Download                                                                                                             |
| :------: | :--: | :---: | :-----: | :------: | :------------: | :----: | :------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  VGG16   | 300  | caffe |  120e   |   9.9    |      43.7      |  25.5  | [config](./ssd300_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428.log.json) |
|  VGG16   | 512  | caffe |  120e   |   19.4   |      30.7      |  29.5  | [config](./ssd512_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849-0a47a1ca.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20210803_022849.log.json) |


| Model name     | Backbone      | Style      | Size        | Speed (ms) | COCO mAP | Outputs | Training Resources |
|:---------------:|:--------------------:|:--------------------:|:--------------------------------------------:|:------------------------------------------------------------------------: | :--------: | :----------: | :--------------------: |
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)   |  MobileNet v2| Tensorflow 2 |320x320                            |19         | 20.2           | Boxes | TPU-8 |
[SSD] (http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz)|MobileNet V1 FPN | Tensorflow 2 |640x640|                        | 48        | 29.1           | Boxes | TPU-8|
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz)| MobileNet V2 FPNLite | Tensorflow|320x320  | 22         | 22.2           | Boxes |
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz) |MobileNet V2 FPNLite | Tensorflow|640x640   | 39         | 28.2           | Boxes |
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) | ResNet50 V1 FPN (RetinaNet50) | Tensorflow|640x640 | 46         | 34.3           | Boxes | 
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)| ResNet50 V1 FPN(RetinaNet50) | Tensorflow |1024x1024                       | 87         | 38.3           | Boxes |
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz)| ResNet101 V1 FPN (RetinaNet101)| Tensorflow |640x640        | 57         | 35.6           | Boxes
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz) | ResNet101 V1 FPN (RetinaNet101)|Tensorflow  |1024x1024             | 104        | 39.5           | Boxes |
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz)| ResNet152 V1 FPN (RetinaNet152) | Tensorflow |640x640                         | 80         | 35.4           | Boxes |
[SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)  | ResNet152 V1 FPN (RetinaNet152) |Tensorflow|1024x1024|                   | 111        | 39.6           | Boxes |

## Results and models of SSD-Lite

|  Backbone   | Size | Training from scratch | Lr schd | Mem (GB) | Inf time (fps) | box AP |                           Config                           |                                                                                                                                                                 Download                                                                                                                                                                 |
| :---------: | :--: | :-------------------: | :-----: | :------: | :------------: | :----: | :--------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MobileNetV2 | 320  |          yes          |  600e   |   4.0    |      69.9      |  21.3  | [config](./ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627.log.json) |

## Notice

### Compatibility

In v2.14.0, [PR5291](https://github.com/open-mmlab/mmdetection/pull/5291) refactored SSD neck and head for more
flexible usage. If users want to use the SSD checkpoint trained in the older versions, we provide a scripts
`tools/model_converters/upgrade_ssd_version.py` to convert the model weights.

```bash
python tools/model_converters/upgrade_ssd_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH}

```

- OLD_MODEL_PATH: the path to load the old version SSD model.
- NEW_MODEL_PATH: the path to save the converted model weights.

### SSD-Lite training settings

There are some differences between our implementation of MobileNetV2 SSD-Lite and the one in [TensorFlow 1.x detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) .

1. Use 320x320 as input size instead of 300x300.
2. The anchor sizes are different.
3. The C4 feature map is taken from the last layer of stage 4 instead of the middle of the block.
4. The model in TensorFlow1.x is trained on coco 2014 and validated on coco minival2014, but we trained and validated the model on coco 2017. The mAP on val2017 is usually a little lower than minival2014 (refer to the results in TensorFlow Object Detection API, e.g., MobileNetV2 SSD gets 22 mAP on minival2014 but 20.2 mAP on val2017).

## Citation

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```
