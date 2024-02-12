# Mask R-CNN( DETECTRON 2 )

Mask R-CNN extends Faster R-CNN to solve **instance segmentation tasks**. It achieves this by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. In principle, Mask R-CNN is an intuitive extension of Faster R-CNN, but constructing the mask branch properly is critical for good results.
Most importantly, Faster R-CNN was not designed for pixel-to-pixel alignment between network inputs and outputs. This is evident in how RoIPool, the de facto core operation for attending to instances, performs coarse spatial quantization for feature extraction. To fix the misalignment, Mask R-CNN utilises a simple, quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations.

## Architecture:
        - Convolution
        - RoIAlign
        - RPN
        - Softmax
        - Dense Connections
        - ResNet
        
## Training Data 
        - MS COCO 
        
## Training Resources
        8 NVIDIA V100 GPUs


### COCO Instance Segmentation Baselines with Mask R-CNN
<!--
./gen_html_table.py --config 'COCO-InstanceSegmentation/mask*50*'{1x,3x}'*' 'COCO-InstanceSegmentation/mask*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP mask_AP
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
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml">R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.584</td>
<td align="center">0.110</td>
<td align="center">5.2</td>
<td align="center">36.8</td>
<td align="center">32.2</td>
<td align="center">137259246</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_DC5_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml">R50-DC5</a></td>
<td align="center">1x</td>
<td align="center">0.471</td>
<td align="center">0.076</td>
<td align="center">6.5</td>
<td align="center">38.3</td>
<td align="center">34.2</td>
<td align="center">137260150</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">38.6</td>
<td align="center">35.2</td>
<td align="center">137260431</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml">R50-C4</a></td>
<td align="center">3x</td>
<td align="center">0.575</td>
<td align="center">0.111</td>
<td align="center">5.2</td>
<td align="center">39.8</td>
<td align="center">34.4</td>
<td align="center">137849525</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml">R50-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.470</td>
<td align="center">0.076</td>
<td align="center">6.5</td>
<td align="center">40.0</td>
<td align="center">35.9</td>
<td align="center">137849551</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.261</td>
<td align="center">0.043</td>
<td align="center">3.4</td>
<td align="center">41.0</td>
<td align="center">37.2</td>
<td align="center">137849600</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml">R101-C4</a></td>
<td align="center">3x</td>
<td align="center">0.652</td>
<td align="center">0.145</td>
<td align="center">6.3</td>
<td align="center">42.6</td>
<td align="center">36.7</td>
<td align="center">138363239</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml">R101-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.545</td>
<td align="center">0.092</td>
<td align="center">7.6</td>
<td align="center">41.9</td>
<td align="center">37.3</td>
<td align="center">138363294</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.340</td>
<td align="center">0.056</td>
<td align="center">4.6</td>
<td align="center">42.9</td>
<td align="center">38.6</td>
<td align="center">138205316</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/metrics.json">metrics</a></td>
</tr>
<!-- ROW: mask_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml">X101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.690</td>
<td align="center">0.103</td>
<td align="center">7.2</td>
<td align="center">44.3</td>
<td align="center">39.5</td>
<td align="center">139653917</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/metrics.json">metrics</a></td>
</tr>
</tbody></table>

