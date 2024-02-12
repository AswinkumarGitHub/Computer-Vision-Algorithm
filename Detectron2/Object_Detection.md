[OBJECT DETECTION USING DETECTRON 2](https://paperswithcode.com/lib/detectron2/faster-r-cnn)

[Detectron2 Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

# Detectron2 Model Zoo and Baselines

## Introduction

This file documents a large collection of baselines trained
with detectron2 in Sep-Oct, 2019.
All numbers were obtained on [Big Basin](https://engineering.fb.com/data-center-engineering/introducing-big-basin-our-next-generation-ai-hardware/)
servers with 8 NVIDIA V100 GPUs & NVLink. The speed numbers are periodically updated with latest PyTorch/CUDA/cuDNN versions.
You can access these models from code using [detectron2.model_zoo](https://detectron2.readthedocs.io/modules/model_zoo.html) APIs.

In addition to these official baseline models, you can find more models in [projects/](projects/).

#### How to Read the Tables
* The "Name" column contains a link to the config file. Models can be reproduced using `tools/train_net.py` with the corresponding yaml config file,
  or `tools/lazyconfig_train_net.py` for python config files.
* Training speed is averaged across the entire training.
  We keep updating the speed with latest version of detectron2/pytorch/etc.,
  so they might be different from the `metrics` file.
  Training speed for multi-machine jobs is not provided.
* Inference speed is measured by `tools/train_net.py --eval-only`, or [inference_on_dataset()](https://detectron2.readthedocs.io/modules/evaluation.html#detectron2.evaluation.inference_on_dataset),
  with batch size 1 in detectron2 directly.
  Measuring it with custom code may introduce other overhead.
  Actual deployment in production should in general be faster than the given inference
  speed due to more optimizations.
* The *model id* column is provided for ease of reference.
  To check downloaded file integrity, any model on this page contains its md5 prefix in its file name.
* Training curves and other statistics can be found in `metrics` for each model.

#### Common Settings for COCO Models
* All COCO models were trained on `train2017` and evaluated on `val2017`.
* The default settings are __not directly comparable__ with Detectron's standard settings.
  For example, our default training data augmentation uses scale jittering in addition to horizontal flipping.

  To make fair comparisons with Detectron's settings, see
  [Detectron1-Comparisons](configs/Detectron1-Comparisons/) for accuracy comparison,
  and [benchmarks](https://detectron2.readthedocs.io/notes/benchmarks.html)
  for speed comparison.
* For Faster/Mask R-CNN, we provide baselines based on __3 different backbone combinations__:
  * __FPN__: Use a ResNet+FPN backbone with standard conv and FC heads for mask and box prediction,
    respectively. It obtains the best
    speed/accuracy tradeoff, but the other two are still useful for research.
  * __C4__: Use a ResNet conv4 backbone with conv5 head. The original baseline in the Faster R-CNN paper.
  * __DC5__ (Dilated-C5): Use a ResNet conv5 backbone with dilations in conv5, and standard conv and FC heads
    for mask and box prediction, respectively.
    This is used by the Deformable ConvNet paper.
* Most models are trained with the 3x schedule (~37 COCO epochs).
  Although 1x models are heavily under-trained, we provide some ResNet-50 models with the 1x (~12 COCO epochs)
  training schedule for comparison when doing quick research iteration.

#### ImageNet Pretrained Models

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. The following backbone models are available:

* [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl): converted copy of [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model.
* [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): converted copy of [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) model.
* [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB.
* [R-50.pkl (torchvision)](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl): converted copy of [torchvision's ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model.
  More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).

Note that the above models have __different__ format from those provided in Detectron: we do not fuse BatchNorm into an affine layer.
Pretrained models in Detectron's format can still be used. For example:
* [X-152-32x8d-IN5k.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl):
  ResNeXt-152-32x8d model trained on ImageNet-5k with Caffe2 at FB (see ResNeXt paper for details on ImageNet-5k).
* [R-50-GN.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl):
  ResNet-50 with Group Normalization.
* [R-101-GN.pkl](https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl):
  ResNet-101 with Group Normalization.

These models require slightly different settings regarding normalization and architecture. See the model zoo configs for reference.

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

### COCO Object Detection Baselines

#### Faster R-CNN:
<!--
(fb only) To update the table in vim:
1. Remove the old table: d}
2. Copy the below command to the place of the table
3. :.!bash

./gen_html_table.py --config 'COCO-Detection/faster*50*'{1x,3x}'*' 'COCO-Detection/faster*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml">R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.551</td>
<td align="center">0.102</td>
<td align="center">4.8</td>
<td align="center">35.7</td>
<td align="center">137257644</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml">R50-DC5</a></td>
<td align="center">1x</td>
<td align="center">0.380</td>
<td align="center">0.068</td>
<td align="center">5.0</td>
<td align="center">37.3</td>
<td align="center">137847829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.210</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">37.9</td>
<td align="center">137257794</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml">R50-C4</a></td>
<td align="center">3x</td>
<td align="center">0.543</td>
<td align="center">0.104</td>
<td align="center">4.8</td>
<td align="center">38.4</td>
<td align="center">137849393</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml">R50-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.378</td>
<td align="center">0.070</td>
<td align="center">5.0</td>
<td align="center">39.0</td>
<td align="center">137849425</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.209</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">40.2</td>
<td align="center">137849458</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml">R101-C4</a></td>
<td align="center">3x</td>
<td align="center">0.619</td>
<td align="center">0.139</td>
<td align="center">5.9</td>
<td align="center">41.1</td>
<td align="center">138204752</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml">R101-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.452</td>
<td align="center">0.086</td>
<td align="center">6.1</td>
<td align="center">40.6</td>
<td align="center">138204841</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.286</td>
<td align="center">0.051</td>
<td align="center">4.1</td>
<td align="center">42.0</td>
<td align="center">137851257</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml">X101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.638</td>
<td align="center">0.098</td>
<td align="center">6.7</td>
<td align="center">43.0</td>
<td align="center">139173657</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/metrics.json">metrics</a></td>
</tr>
</tbody></table>

#### RetinaNet:
<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50 R50 R101 --fields lr_sched train_speed inference_speed mem box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml">R50</a></td>
<td align="center">1x</td>
<td align="center">0.205</td>
<td align="center">0.041</td>
<td align="center">4.1</td>
<td align="center">37.4</td>
<td align="center">190397773</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml">R50</a></td>
<td align="center">3x</td>
<td align="center">0.205</td>
<td align="center">0.041</td>
<td align="center">4.1</td>
<td align="center">38.7</td>
<td align="center">190397829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml">R101</a></td>
<td align="center">3x</td>
<td align="center">0.291</td>
<td align="center">0.054</td>
<td align="center">5.2</td>
<td align="center">40.4</td>
<td align="center">190397697</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/metrics.json">metrics</a></td>
</tr>
</tbody></table>


#### RPN & Fast R-CNN:
<!--
./gen_html_table.py --config 'COCO-Detection/rpn*' 'COCO-Detection/fast_rcnn*' --name "RPN R50-C4" "RPN R50-FPN" "Fast R-CNN R50-FPN" --fields lr_sched train_speed inference_speed mem box_AP prop_AR
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">prop.<br/>AR</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: rpn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/rpn_R_50_C4_1x.yaml">RPN R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.130</td>
<td align="center">0.034</td>
<td align="center">1.5</td>
<td align="center"></td>
<td align="center">51.6</td>
<td align="center">137258005</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/model_final_450694.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/metrics.json">metrics</a></td>
</tr>
<!-- ROW: rpn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/rpn_R_50_FPN_1x.yaml">RPN R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.186</td>
<td align="center">0.032</td>
<td align="center">2.7</td>
<td align="center"></td>
<td align="center">58.0</td>
<td align="center">137258492</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/metrics.json">metrics</a></td>
</tr>
<!-- ROW: fast_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml">Fast R-CNN R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.140</td>
<td align="center">0.029</td>
<td align="center">2.6</td>
<td align="center">37.8</td>
<td align="center"></td>
<td align="center">137635226</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/metrics.json">metrics</a></td>
</tr>
</tbody></table>

