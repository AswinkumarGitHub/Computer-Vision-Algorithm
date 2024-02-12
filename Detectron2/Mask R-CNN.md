# Mask R-CNN( DETECTRON 2 )

Mask R-CNN extends Faster R-CNN to solve instance segmentation tasks. It achieves this by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. In principle, Mask R-CNN is an intuitive extension of Faster R-CNN, but constructing the mask branch properly is critical for good results.
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

