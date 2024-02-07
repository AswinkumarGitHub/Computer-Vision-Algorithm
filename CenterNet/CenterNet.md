# CenterNet




## Abstract

CenterNet is an anchorless object detection architecture. It is a one-stage object detector that detects each object as a triplet, rather than a pair, of keypoints. It utilizes two customized modules named cascade corner pooling and center pooling, which play the roles of enriching information collected by both top-left and bottom-right corners and providing more recognizable information at the central regions, respectively. The intuition is that, if a predicted bounding box has a high IoU with the ground-truth box, then the probability that the center keypoint in its central region is predicted as the same class is high, and vice versa. Thus, during inference, after a proposal is generated as a pair of corner keypoints, we determine if the proposal is indeed an object by checking if there is a center keypoint of the same class falling within its central region.

<div align=center>
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-23_at_12.29.21_PM_bAPb2Mm.png" height="300"/>
</div>

## Training Data: COCO
      

##     Training Resource: TPU-8 and TPU-32
      
##     Architecture:
        - 
        - RPN
        - ResNet
        - RoIPool

## Results and Models

| Model name     | Backbone            | Size        | Speed (ms) | COCO mAP | Outputs | Training Resources |
|:---------------:|:--------------------:|:----------------------------------------------------------------:|:------------------------------------------------------------------------: | :--------: | :----------: | :--------------------: |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz) | [HourGlass104] | [512x512]                 | 70         | 41.9           | Boxes |  TPU-8 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz)  | [HourGlass104_ Keypoints] | [512x512]   | 76         | 40.0/61.4           | Boxes/Keypoints |  TPU-32 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_1024x1024_coco17_tpu-32.tar.gz) | [HourGlass104] | [1024x1024] | 197       | 44.5           | Boxes |  TPU-32 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_kpts_coco17_tpu-32.tar.gz) | [HourGlass104_ Keypoints] | [1024x1024] |  211       | 42.8/64.5          | Boxes/Keypoints |  TPU-32 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz) | [Resnet50 V1 FPN ] | [512x512] | 27         | 31.2           | Boxes|  TPU-8 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.tar.gz) | [Resnet50 V1 FPN_Keypoints ] | [512x512] | 27         | 31.2           | Boxes|  TPU-8 |
|[CenterNet ](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz) | Resnet101 V1 FPN | 512x512 | 34 | 34.2 | Boxes |TPU-8 |
|[CenterNet ](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz) | Resnet50 V2 | 512x512 | 27 | 29.5 | Boxes |TPU-8 |
|[CenterNet ](http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_kpts_coco17_tpu-8.tar.gz) | Resnet50 V2 | 512x512 | 30 | 27.6/48.2 | Boxes/Keypoints |TPU-8 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz) | MobileNetV2 FPN | 512x512 | 6 | 23.4 | Boxes |TPU-8 |
|[CenterNet](http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_kpts.tar.gz) | MobileNetV2 FPN | 512x512 | 6 | 41.7 | Keypoints |TPU-8 |
