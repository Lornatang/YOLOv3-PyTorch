# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydet.config import parse_model_config
from .activition import HSigmoid
from .activition import HSwish
from .activition import Mish
from .activition import Swish
from .conv import MixConv2d
from .conv import SeModule
from .res import InvertedResidual
from ..common import ONNX_EXPORT
from ..common import model_info
from ..concat import FeatureConcat
from ..fuse import WeightFeatureFusion
from ..fuse import fuse_conv_and_bn


def create_modules(module_defines, image_size):
    # Constructs module list of layer blocks from module configuration in module_defines
    # expand if necessary
    image_size = [image_size] * 2 if isinstance(image_size, int) else image_size
    _ = module_defines.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, module in enumerate(module_defines):
        modules = nn.Sequential()

        if module["type"] == "convolutional":
            bn = module["batch_normalize"]
            in_channels = module["in_features"] if "in_features" in module else output_filters[-1]
            filters = module["filters"]
            size = module["size"]
            stride = module["stride"] if "stride" in module else (
                module["stride_y"], module["stride_x"])
            groups = module["groups"] if "groups" in module else 1
            if isinstance(size, int):  # single-size conv
                modules.add_module("Conv2d", nn.Conv2d(in_channels=in_channels,
                                                       out_channels=filters,
                                                       kernel_size=size,
                                                       stride=stride,
                                                       padding=(size - 1) // 2
                                                       if module["pad"] else 0,
                                                       groups=groups,
                                                       bias=not bn))
            else:  # multiple-size conv
                modules.add_module("MixConv2d", MixConv2d(in_channels=in_channels,
                                                          out_channels=filters,
                                                          kernel_size=size,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(num_features=filters,
                                                                 momentum=0.03,
                                                                 eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if module["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, True))
            elif module["activation"] == "relu":
                modules.add_module("activation", nn.ReLU(inplace=True))
            elif module["activation"] == "relu6":
                modules.add_module("activation", nn.ReLU6(inplace=True))
            elif module["activation"] == "swish":
                modules.add_module("activation", Swish())
            elif module["activation"] == "mish":
                modules.add_module("activation", Mish())
            elif module["activation"] == "hswish":
                modules.add_module("activation", HSwish())
            elif module["activation"] == "hsigmoid":
                modules.add_module("activation", HSigmoid())

        elif module["type"] == "BatchNorm2d":
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif module["type"] == "maxpool":
            size = module["size"]
            stride = module["stride"]
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=(size - 1) // 2)
            if size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module("MaxPool2d", maxpool)
            else:
                modules = maxpool

        elif module["type"] == "avgpool":
            size = module["size"]
            stride = module["stride"]
            modules.add_module("AvgPool2d", nn.AvgPool2d(kernel_size=size, stride=stride,
                                                         padding=(size - 1) // 2))

        elif module["type"] == "semodule":
            in_channels = module["in_features"]
            modules.add_module("SeModule", SeModule(in_channels))

        elif module["type"] == "InvertedResidual":
            in_channels = module["in_channels"]
            out_channels = module["out_channels"]
            stride = module["stride"]
            modules.add_module("InvertedResidual", InvertedResidual(in_channels=in_channels,
                                                                    out_channels=out_channels,
                                                                    stride=stride).cuda())

        elif module["type"] == "dense":
            bn = module["batch_normalize"]
            in_features = module["in_features"]
            out_features = module["out_features"]
            modules.add_module("Linear", nn.Linear(in_features=in_features,
                                                   out_features=out_features,
                                                   bias=not bn))

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(num_features=out_features,
                                                                 momentum=0.003,
                                                                 eps=1E-4))

        elif module["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                # image_size = (320, 192)
                modules = nn.Upsample(
                    size=tuple(int(x * g) for x in image_size))
            else:
                modules = nn.Upsample(scale_factor=module["stride"])

        # nn.Sequential() placeholder for "route" layer
        elif module["type"] == "route":
            layers = module['layers']
            filters = sum([output_filters[layer + 1 if layer > 0 else layer]
                           for layer in layers])
            routs.extend([i + layer if layer < 0 else layer
                          for layer in layers])
            modules = FeatureConcat(layers=layers)

        # nn.Sequential() placeholder for "shortcut" layer
        elif module["type"] == "shortcut":
            layers = module["from"]
            filters = output_filters[-1]
            routs.extend([i + layer if layer < 0 else layer
                          for layer in layers])
            modules = WeightFeatureFusion(layers=layers,
                                          weight='weights_type' in module)

        elif module["type"] == "yolo":
            yolo_index += 1
            stride = [32, 16, 8, 4, 2][yolo_index]  # P3-P7 stride
            layers = module["from"] if "from" in module else []
            modules = YOLOLayer(anchors=module["anchors"][module["mask"]],
                                # anchor list
                                nc=module["classes"],  # number of classes
                                image_size=image_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride)

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                bo = -4.5  #  obj bias
                # cls bias: class probability is sigmoid(p) = 1/nc
                bc = math.log(1 / (modules.nc - 0.99))

                j = layers[yolo_index] if "from" in module else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                # shape(3,85)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)
                bias[:, 4] += bo - bias[:, 4].mean()  # obj
                # cls, view with utils.print_model_biases(model)
                bias[:, 5:] += bc - bias[:, 5:].mean()
                module_list[j][0].bias = torch.nn.Parameter(
                    bias_, requires_grad=bias_.requires_grad)
            except:
                print("WARNING: smart bias initialization failure.")

        else:
            print(f"Warning: Unrecognized Layer Type: {module['type']}")

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, config, image_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defines = parse_model_config(config)
        self.module_list, self.routs = create_modules(self.module_defines,
                                                      image_size)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        # (int32) version info: major, minor, revision
        self.version = np.array([0, 2, 5], dtype=np.int32)
        # (int64) number of images seen during training
        self.seen = np.array([0], dtype=np.int64)
        self.info()  # print model description

    def forward(self, x, augment=False):

        if not augment:
            return self.forward_once(x)
        else:
            image_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            flip_lr_image = scale_image(x.flip(3), s[0], same_shape=False)  # flip-lr and scale
            scale = scale_image(x, s[1], same_shape=False)  # scale
            for i, xi in enumerate((x, flip_lr_image, scale)):
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = image_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False):
        image_size = x.shape[-2:]
        yolo_out, out = [], []

        # Augment images (inference and test only)
        s = [0.83, 0.67]  # scales
        if augment:
            flip_lr_image = scale_image(x.flip(3), s[0])  # flip-lr and scale
            scale = scale_image(x, s[1])  # scale
            x = torch.cat((x, flip_lr_image, scale), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x, image_size, out))
            else:
                x = module(x)
            out.append(x if self.routs[i] else [])

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, x.shape[0], dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = image_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print("Fusing Conv2d() and BatchNorm2d() layers...")
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info()  # yolov3-spp reduced from 225 to 152 layers

    def info(self):
        model_info(self)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, image_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny = 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            # number x, y grid points
            self.create_grids((image_size[1] // stride, image_size[0] // stride))

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.Tensor(ng).to(device)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, pred, image_size, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            pred = out[self.layers[i]]
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), pred.device)

            # outputs and weights
            # sigmoid weights (faster)
            w = torch.sigmoid(pred[:, -n:]) * (2 / n)

            # weighted ASFF sum
            pred = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    pred += w[:, j:j + 1] * F.interpolate(
                        input=out[self.layers[j]][:, :-n],
                        size=[ny, nx],
                        mode="bilinear",
                        align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = pred.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), pred.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        pred = pred.view(bs, self.na, self.no, self.ny, self.nx).permute(
            0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return pred

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1 / self.ng.repeat((m, 1))
            grid = self.grid.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(
                (1, 1, self.nx, self.ny, 1)).view(m, 2) * ng

            pred = pred.view(m, self.no)
            xy = torch.sigmoid(pred[:, 0:2]) + grid  # x, y
            wh = torch.exp(pred[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(
                pred[:, 4:5]) if self.nc == 1 else torch.sigmoid(
                pred[:, 5:self.no]) * torch.sigmoid(pred[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = pred.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            # wh yolo method
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            # view [1, 3, 13, 13, 85] as [1, 507, 85]
            return io.view(bs, -1, self.no), pred


def get_yolo_layers(model):
    # [82, 94, 106] for yolov3
    return [i for i, x in enumerate(model.module_defines) if x["type"] == "yolo"]
