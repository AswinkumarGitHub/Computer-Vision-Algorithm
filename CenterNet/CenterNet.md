# CenterNet




## Abstract

CenterNet is an anchorless object detection architecture. This structure has an important advantage in that it replaces the classical NMS (Non Maximum Suppression) at the post process, with a much more elegant algorithm, that is natural to the CNN flow. This mechanism enables a much faster inference.

<div align=center>
<img src="https://miro.medium.com/v2/resize:fit:1400/1*y82flEmdWr20NjuevgQ8-Q.png" height="300"/>
</div>

## Training Data: COCO
      

##     Training Resource: TPU-8 and TPU-32
      
##     Architecture:
        - FPN
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
