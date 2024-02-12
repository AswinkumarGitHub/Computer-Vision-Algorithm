# Faster R-CNN (R101-C4, 3x)
|Parameters | FLOPs    |    File Size | Training Data| Training Resources | Training Time |
|:-----------:| :-----------:| :-------------:|:-----------:|:----------:|:----------:|:-------:|
|53 Million | 888 Billion | 202.02 MB | MS COCO | 8 NVIDIA V100 GPUs| 1.93 days |






## Architecture:
        - Convolution
        - RoIPool
        - RPN
        - Softmax
        - ResNet


Architecture	
ID	138204752
Max Iter	270000
lr sched	3x
FLOPs Input No	100
Backbone Layers	101
train time (s/iter)	0.619
Training Memory (GB)	5.9
inference time (s/im)	0.139
