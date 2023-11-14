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
pip3 install -e .
```

## All pretrained model weights

- [Google Driver](https://drive.google.com/drive/folders/1b5f3FSeZwIFs4bp17OWKhQeaEcMKJyma?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1GvepU_8APWChG_03yUVQ_w?pwd=7e0g)

## Inference (e.g YOLOv3_Tiny-VOC0712)

```shell
# Download YOLOv3_Tiny-VOC0712 model weights to `./results/pretrained_models`
wget https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-VOC0712-20231107.pth.tar -O ./resutls/pretrained_models/YOLOv3_Tiny-VOC0712-20231107.pth.tar
python3 ./tools/inference.py ./data/examples/dog.jpg
# Result will be saved to `./results/predict/YOLOv3_Tiny-VOC0712/dog.jpg`
```

<div align="center">
<img src="figure/dog.jpg" width="768">
</div>

## Test

### VOC0712

```shell
# Download dataset to `./data`
cd ./scripts
bash ./process_voc0712_dataset.sh
cd ..
# Download pretrained model weights to `./results/pretrained_models`
wget https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-VOC0712-20231107.pth.tar -O ./resutls/pretrained_models/YOLOv3_Tiny-VOC0712-20231107.pth.tar
python3 ./tools/test.py ./configs/YOLOv3_Tiny-VOC0712.yaml
```

### Results

#### COCO2014

|                                                                     Model                                                                     | Size | mAP<sup>val<br/>0.5:0.95 | FLOPs(G) | Parameters(M) | Memory(MB) |
|:---------------------------------------------------------------------------------------------------------------------------------------------:|:----:|:------------------------:|:--------:|:-------------:|:----------:|
|     [**YOLOv3_Tiny-COCO2014**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-COCO2014-20231107.pth.tar)     | 416  |           18.7           |   5.6    |     0.71      |    8.9     |
| [**YOLOv3_Tiny_PRN-COCO2014**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny_PRN-COCO2014-20231107.pth.tar) | 416  |           11.1           |   3.5    |     0.66      |    4.9     |
|          [**YOLOv3-COCO2014**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3-COCO2014-20231107.pth.tar)          | 416  |           66.7           |   66.2   |     0.88      |    61.9    |
|      [**YOLOv3_SPP-COCO2014**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_SPP-COCO2014-20231107.pth.tar)      | 416  |           66.7           |   66.5   |     0.88      |    63.0    |

#### VOC

|                                                                             Model                                                                             | Size | mAP<sup>val<br/>0.5:0.95 | FLOPs(B) | Memory(MB) | Parameters(M) |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----:|:------------------------:|:--------:|:----------:|:-------------:|
|              [**YOLOv3_Tiny-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-VOC0712-20231107.pth.tar)              | 416  |           58.8           |   5.5    |    0.27    |      8.7      |
|          [**YOLOv3_Tiny_PRN-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny_PRN-VOC0712-20231107.pth.tar)          | 416  |           47.9           |   3.5    |    0.27    |      4.9      |
|                   [**YOLOv3-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3-VOC0712-20231107.pth.tar)                   | 416  |           82.9           |   65.7   |    0.61    |     61.6      |
|               [**YOLOv3_SPP-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_SPP-VOC0712-20231107.pth.tar)               | 416  |           83.2           |   66.1   |    0.88    |     62.7      |
|       [**YOLOv3_MobileNetV1-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_MobileNetV1-VOC0712-20231107.pth.tar)       | 416  |           65.6           |   6.6    |    0.69    |      6.2      |
|       [**YOLOv3_MobileNetV2-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_MobileNetV2-VOC0712-20231107.pth.tar)       | 416  |           68.2           |   3.5    |    0.49    |      4.3      |
| [**YOLOv3_MobileNetV3_Large-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_MobileNetV3_Large-VOC0712-20231107.pth.tar) | 416  |           70.1           |   2.8    |    0.50    |      4.7      |
| [**YOLOv3_MobileNetV3_Small-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_MobileNetV3_Small-VOC0712-20231107.pth.tar) | 416  |           53.7           |   1.5    |    0.48    |      2.8      |
|             [**YOLOv3_VGG16-VOC0712**](https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_VGG16-VOC0712-20231107.pth.tar)             | 416  |           74.1           |  122.8   |    0.74    |     35.5      |

## Train

### VOC0712

```shell
# Download dataset to `./data`
cd ./scripts
bash ./process_voc0712_dataset.sh
cd ..
# Download pretrained model weights to `./results/pretrained_models`
wget https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-VOC0712-20231107.pth.tar -O ./resutls/pretrained_models/YOLOv3_Tiny-VOC0712-20231107.pth.tar
python3 ./tools/train.py ./configs/YOLOv3_Tiny-VOC0712.yaml
```

### COCO2014 & COCO2017

```shell
# COCO2014
# Download dataset to `./data`
cd ./scripts
bash ./process_coco2014_dataset.sh
cd ..
# Download pretrained model weights to `./results/pretrained_models`
wget https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-COCO2014-20231107.pth.tar -O ./resutls/pretrained_models/YOLOv3_Tiny-COCO2014-20231107.pth.tar
python3 ./tools/train.py ./configs/YOLOv3_Tiny-COCO2014.yaml

# COCO2017
# Download dataset to `./data`
cd ./scripts
bash ./process_coco2017_dataset.sh
cd ..
# Download pretrained model weights to `./results/pretrained_models`
wget https://github.com/Lornatang/YOLOv3-PyTorch/releases/download/0.1.5/YOLOv3_Tiny-COCO2017-20231107.pth.tar -O ./resutls/pretrained_models/YOLOv3_Tiny-COCO2017-20231107.pth.tar
python3 ./tools/train.py ./configs/YOLOv3_Tiny-COCO2017.yaml
```

### Custom dataset

Details see [CustomDataset.md](./data/README.md).

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
