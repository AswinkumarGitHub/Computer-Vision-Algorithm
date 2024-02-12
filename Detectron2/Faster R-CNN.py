# !python -m pip install pyyaml==5.1
# # Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# # so we install from source instead. This takes a few minutes.
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # Install pre-built detectron2 that matches pytorch version, if released:
# # See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# #!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html

# # exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime

# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# # (add --user if you don't have permission)

# # Or, to install it from a local clone:
# git clone https://github.com/facebookresearch/detectron2.git
# python -m pip install -e detectron2

# # Or if you are on macOS
# CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from detectron2 import model_zoo


# get image
#!wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
im = cv2.imread("demo.png")

# Create config
#We are using the pre-trained Detectron2 model, as shown below.
cfg = get_cfg()

cfg.MODEL.DEVICE = "cpu"
# load the pre trained model from Detectron2 model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# set confidence threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
# load model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")


predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
output_image = v.get_image()[:, :, ::-1]  # Convert image back to OpenCV format (BGR)
cv2.imshow("Predictions", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
