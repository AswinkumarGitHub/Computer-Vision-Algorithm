# Faster R-CNN

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

<!-- [ALGORITHM] -->

## Abstract

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks.


<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143881188-ab87720f-5059-4b4e-a928-b540fb8fb84d.png" height="300"/>
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
