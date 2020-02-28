# Introduction

This directory contains PyTorch YOLOv3 software developed by Ultralytics LLC.

# Description

The repo contains inference and training code for YOLOv3 in PyTorch. The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO:** https://pjreddie.com/darknet/yolo/.

# Requirements

Python 3.7 or later with all of the `pip install -U -r requirements.txt` packages including:
- `torch >= 1.4`
- `opencv-python`
- `Pillow`

All dependencies are included in the associated docker images. Docker requirements are: 
- Nvidia Driver >= 440.44
- Docker Engine - CE >= 19.03

# Jupyter Notebook

Our Jupyter [notebook](https://colab.research.google.com/github/ultralytics/yolov3/blob/master/examples.ipynb) provides quick training, inference and testing examples.

# Training

**Start Training:** `python3 train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Each epoch trains on 117,263 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set.

**Resume Training:** `python3 train.py --resume` to resume training from `weights/last.pt`.

**Plot Training:** `from utils import utils; utils.plot_results()` plots training results from `coco_16img.data`, `coco_64img.data`, 2 example datasets available in the `data/` folder, which train and test on the first 16 and 64 images of the COCO2014-trainval dataset.


## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

Augmentation | Description
--- | ---
Translation | +/- 10% (vertical and horizontal)
Rotation | +/- 5 degrees
Shear | +/- 2 degrees (vertical and horizontal)
Scale | +/- 10%
Reflection | 50% probability (horizontal-only)
H**S**V Saturation | +/- 50%
HS**V** Intensity | +/- 50%

# Inference

`detect.py` runs inference on any sources:

```bash
python3 detect.py --source ...
```

- Image:  `--source file.jpg`
- Video:  `--source file.mp4`
- Directory:  `--source dir/`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

To run a specific models:

**YOLOv3:** `python3 detect.py --cfg cfg/yolov3.cfg --weights yolov3.weights`  

**YOLOv3-tiny:** `python3 detect.py --cfg cfg/yolov3-tiny.cfg --weights yolov3-tiny.weights`  

**YOLOv3-SPP:** `python3 detect.py --cfg cfg/yolov3-spp.cfg --weights yolov3-spp.weights`  


### Train on Custom Dataset
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```bash
# move to config dir
cd cfg/
# create custom model 'yolov3-custom.cfg'. (In fact, it is OK to modify two lines of parameters, see `create_model.sh`)                              
sh ./create_model.sh <num-classes>
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```bash
python3 train.py --cfg cfg/yolov3-custom.cfg --data cfg/custom.data --epochs 100 --multi-scale
```

Add `--weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.

## Darknet Conversion

```bash
$ git clone https://github.com/ultralytics/yolov3 && cd yolov3

# convert darknet cfg/weights to pytorch model
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')"
Success: converted 'weights/yolov3-spp.weights' to 'converted.pt'

# convert cfg/pytorch model to darknet weights
$ python3  -c "from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.pt')"
Success: converted 'weights/yolov3-spp.pt' to 'converted.weights'
```

# mAP

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt
```

- mAP@0.5 run at `--iou-thr 0.5`, mAP@0.5...0.95 run at `--iou-thr 0.7`
- Darknet results: https://arxiv.org/abs/1804.02767

<i></i>                      |Size |COCO mAP<br>@0.5...0.95 |COCO mAP<br>@0.5 
---                          | ---         | ---         | ---
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |320 |14.0<br>28.7<br>30.5<br>**36.4** |29.1<br>51.8<br>52.3<br>**55.7**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |416 |16.0<br>31.2<br>33.9<br>**40.0** |33.0<br>55.4<br>56.9<br>**60.0**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |512 |16.6<br>32.7<br>35.6<br>**41.5** |34.9<br>57.7<br>59.5<br>**61.4**
YOLOv3-tiny<br>YOLOv3<br>YOLOv3-SPP<br>**[YOLOv3-SPP-ultralytics](https://drive.google.com/open?id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4)** |608 |16.6<br>33.1<br>37.0<br>**41.9** |35.4<br>58.2<br>60.7<br>**61.6**

```bash
$ python3 test.py --cfg yolov3-spp.cfg --weights yolov3-spp-ultralytics.pt --img 608
```

```text
Namespace(batch_size=32, cfg='yolov3-spp.cfg', conf_thres=0.001, data='data/coco2014.data', device='', img_size=608, iou_thres=0.6, save_json=True, single_cls=False, task='test', weights='yolov3-spp-ultralytics.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla T4', total_memory=15079MB)

               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 157/157 [04:25<00:00,  1.01s/it]
                 all     5e+03  3.51e+04    0.0453     0.885     0.609    0.0852
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.462
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.436
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.659
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.741
```

# References and references
This project is based on [yolov3](https://github.com/ultralytics/yolov3) re-development.
