<div align="center">
<img src="figure/dog.jpg" width="400px">
</div>

# YOLOv3-PyTorch

## Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [From PyPI](#from-pypi)
    - [Local Install](#local-install)
- [Inference (TODO)](#inference-todo)
- [All pretrained model weights](#all-pretrained-model-weights)
- [How Test and Train](#how-test-and-train)
    - [Test yolov3_tiny_voc model](#test-yolov3tinyvoc-model)
    - [Train yolov3_tiny_voc model](#train-yolov3tinyvoc-model)
    - [Resume train yolov3_tiny_voc model](#resume-train-yolov3tinyvoc-model)
- [Result](#result)
- [Contributing](#contributing)
- [Credit](#credit)
    - [YOLOv3: An Incremental Improvement](#yolov3--an-incremental-improvement)

## Introduction

This repository contains an op-for-op PyTorch reimplementation of [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767v1.pdf).

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.0.0+
- CUDA 11.8+
- Ubuntu 22.04+

### From PyPI

```bash
pip3 install yolov3_pytorch -i https://pypi.org/simple
```

### Local Install

```bash
git clone https://github.com/Lornatang/YOLOv3-PyTorch.git
cd YOLOv3-PyTorch
pip3 install -r requirements.txt
python3 setup.py install
```

## Inference (e.g YOLOv3_tiny-VOC)

```bash
# Download YOLOv3_tiny-VOC model weights to `./results/pretrained_models`
wget https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_tiny-VOC-20200402.pth.tar ./resutls/pretrained_models/YOLOv3_tiny-VOC-20200402.pth.tar
python3 ./tools/detect.py
```

## All pretrained model weights

- [Google Driver](https://drive.google.com/drive/folders/1b5f3FSeZwIFs4bp17OWKhQeaEcMKJyma?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1GvepU_8APWChG_03yUVQ_w?pwd=7e0g)


## How Test and Train

Both training and testing only need to modify the `train_config.py` or `test_config.py` file.

### Test yolov3_tiny_voc model

Modify the `test_config.py` file.

- line 18: `model_arch_name` change to `yolov3_tiny_voc`.
- line 34: `test_dataset_config_path` change to `./data/voc.data`.
- line 38: `model_weights_path` change to `./results/pretrained_models/YOLOv3_tiny-COCO.weights`.

```bash
python3 test.py
```

### Train yolov3_tiny_voc model

Modify the `train_config.py` file.

- line 18: `model_arch_name` change to `yolov3_tiny_voc`.
- line 58: `upscale_factor` change to `./data/voc.data`.

```bash
python3 train.py
```

### Resume train yolov3_tiny_voc model

Modify the `train_config.py` file.

- line 18: `model_arch_name` change to `yolov3_tiny_voc`.
- line 58: `upscale_factor` change to `./data/voc.data`.
- line 74: `resume_model_weights_path` change to `f"./samples/YOLOv3_tiny-VOC0712/epoch_xxx.pth.tar"`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1804.02767v1.pdf](https://arxiv.org/pdf/1804.02767v1.pdf)

In the following table, the mAP value in `()` indicates the result of the project, and `-` indicates no test.

|         Model         |   Train dataset   | Test dataset | Size |     mAP     | 
|:---------------------:|:-----------------:|:------------:|:----:|:-----------:|
|  yolov3_tiny_prn_voc  | VOC07+12 trainval |  VOC07 test  | 416  | -(**56.4**) |
|    yolov3_tiny_voc    | VOC07+12 trainval |  VOC07 test  | 416  | -(**58.8**) |
|      yolov3_voc       | VOC07+12 trainval |  VOC07 test  | 416  | -(**79.0**) |
|    yolov3_spp_voc     | VOC07+12 trainval |  VOC07 test  | 416  | -(**75.3**) |
|    mobilenetv1_voc    | VOC07+12 trainval |  VOC07 test  | 416  | -(**66.0**) |
|    mobilenetv2_voc    | VOC07+12 trainval |  VOC07 test  | 416  | -(**69.3**) |
| mobilenetv3_small_voc | VOC07+12 trainval |  VOC07 test  | 416  | -(**53.8**) |
| mobilenetv3_large_voc | VOC07+12 trainval |  VOC07 test  | 416  | -(**71.1**) |
|      alexnet_voc      | VOC07+12 trainval |  VOC07 test  | 416  | -(**56.4**) |
|       vgg16_voc       | VOC07+12 trainval |  VOC07 test  | 416  | -(**74.5**) |

```bash
# Download `YOLOv3_tiny-VOC0712-d24f2c25.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./detect.py
```

Output1:

<span align="center"><img width="768" height="576" src="figure/dog.jpg"/></span>

Output2:

<span align="center"><img width="640" height="424" src="figure/person.jpg"/></span>

```text
Loaded `` pretrained model weights successfully.
image 1/2 data/examples/dog.jpg: 480x608 1 bicycle, 1 car, 1 dog, 
image 2/2 data/examples/person.jpg: 416x608 1 dog, 1 person, 1 sheep,
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### YOLOv3: An Incremental Improvement

_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though,
don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at
the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5
AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online
at https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```bibtex
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
