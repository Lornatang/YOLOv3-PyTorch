# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
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
from typing import Any, List, Dict

import numpy as np
import torch
from torch import nn, Tensor
from torchvision.ops.misc import SqueezeExcitation

from .module import MixConv2d, InvertedResidual, WeightedFeatureFusion, FeatureConcat, YOLOLayer, fuse_conv_and_bn, make_divisible, scale_img

__all__ = [
    "Darknet"
]


class Darknet(nn.Module):
    def __init__(
            self,
            model_config_path: str,
            img_size: tuple = (416, 416),
            gray: bool = False,
            compile_mode: bool = False,
            onnx_export: bool = False,
    ) -> None:
        """

        Args:
            self.model_config_path_path (str): Model configuration file path.
            img_size (tuple, optional): Image size. Default: (416, 416).
            gray (bool, optional): Whether to use grayscale imgs. Default: ``False``.
            compile_mode (bool, optional): PyTorch 2.0 supports model compilation, the compiled model will have a prefix than
                the original model parameters, default: ``False``.
            onnx_export (bool, optional): Whether to export to onnx. Default: ``False``.

        """
        super(Darknet, self).__init__()
        self.model_config_path = model_config_path
        self.img_size = img_size
        self.gray = gray
        self.compile_mode = compile_mode
        self.onnx_export = onnx_export

        self.module_defines = self.create_module_defines()
        self.module_list, self.routs = self.create_module_list()
        self.yolo_layers = self.get_yolo_layers()

        # Obj losses
        self.giou_ratio = 1.0

        # Darknet parameters
        self.version = np.array([0, 1, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def create_module_defines(self) -> List[Dict[str, Any]]:
        """Parses the yolo-v3 layer configuration file and returns module definitions.

        Returns:
            module_define (List[Dict[str, Any]]): module definitions
        """

        with open(self.model_config_path, "r") as f:
            lines = f.read().split("\n")
        lines = [x for x in lines if x and not x.startswith("#")]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        module_defines = []  # module definitions
        for line in lines:
            if line.startswith("["):  # This marks the start of a new block
                module_defines.append({})
                module_defines[-1]["type"] = line[1:-1].rstrip()
                if module_defines[-1]["type"] == "convolutional":
                    module_defines[-1]["batch_normalize"] = 0  # pre-populate with zeros (may be overwritten later)
            else:
                key, val = line.split("=")
                key = key.rstrip()

                if key == "anchors":  # return nparray
                    module_defines[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
                elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):  # return array
                    module_defines[-1][key] = [int(x) for x in val.split(",")]
                else:
                    val = val.strip()
                    if val.isnumeric():  # return int or float
                        module_defines[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                    else:
                        module_defines[-1][key] = val  # return string

        # Check all fields are supported
        supported = ["type", "in_channels", "out_channels", "in_features", "out_features",
                     "num_features", "batch_normalize", "filters", "size", "stride", "pad", "activation",
                     "layers", "groups", "from", "mask", "anchors", "classes", "num", "jitter",
                     "ignore_thresh", "truth_thresh", "random", "stride_x", "stride_y", "weights_type",
                     "weights_normalization", "scale_x_y", "beta_nms", "nms_kind", "iou_loss", "padding",
                     "iou_normalizer", "cls_normalizer", "iou_thresh", "expand_size", "squeeze_excitation"]

        f = []  # fields

        for x in module_defines[1:]:
            [f.append(k) for k in x if k not in f]

        u = [x for x in f if x not in supported]  # unsupported fields

        assert not any(u), f"Unsupported fields {self.model_config_path_path}"

        return module_defines

    def create_module_list(self) -> [nn.ModuleList, list]:
        """Constructs module list of layer blocks from module configuration in module_define

        Returns:
            module_define (nn.ModuleList): Module list
            routs_binary (list): Hyper-parameters

        """
        img_size = [self.img_size] * 2 if isinstance(self.img_size, int) else self.img_size  # expand if necessary
        _ = self.module_defines.pop(0)  # cfg training hyper-params (unused)
        output_filters = [3] if not self.gray else [1]
        module_list = nn.ModuleList()
        routs = []  # list of layers which rout to deeper layers
        yolo_index = -1
        i = 0
        filters = 3

        for i, module in enumerate(self.module_defines):
            modules = nn.Sequential()

            if module["type"] == "convolutional":
                bn = module["batch_normalize"]
                filters = module["filters"]
                k = module["size"]  # kernel size
                stride = module["stride"] if "stride" in module else (module["stride_y"], module["stride_x"])
                if isinstance(k, int):  # single-size conv
                    modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                           out_channels=filters,
                                                           kernel_size=k,
                                                           stride=stride,
                                                           padding=k // 2 if module["pad"] else 0,
                                                           groups=module["groups"] if "groups" in module else 1,
                                                           bias=not bn))
                else:  # multiple-size conv
                    modules.add_module("MixConv2d", MixConv2d(in_channels=output_filters[-1],
                                                              out_channels=filters,
                                                              kernel_size_tuple=k,
                                                              stride=stride,
                                                              bias=not bn))

                if bn:
                    modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
                else:
                    routs.append(i)  # detection output (goes into yolo layer)

                if module["activation"] == "leaky":
                    modules.add_module("activation", nn.LeakyReLU(0.1, True))
                elif module["activation"] == "relu":
                    modules.add_module("activation", nn.ReLU(True))
                elif module["activation"] == "relu6":
                    modules.add_module("activation", nn.ReLU6(True))
                elif module["activation"] == "mish":
                    modules.add_module("activation", nn.Mish(True))
                elif module["activation"] == "hard_swish":
                    modules.add_module("activation", nn.Hardswish(True))
                elif module["activation"] == "hard_sigmoid":
                    modules.add_module("activation", nn.Hardsigmoid(True))

            elif module["type"] == "BatchNorm2d":
                filters = output_filters[-1]
                modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4)
                if i == 0 and filters == 3:  # normalize RGB img
                    modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                    modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

            elif module["type"] == "maxpool":
                k = module["size"]  # kernel size
                stride = module["stride"]
                maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
                if k == 2 and stride == 1:  # yolov3_pytorch-tiny
                    modules.add_module("ZeroPad2d", nn.ZeroPad2d((0, 1, 0, 1)))
                    modules.add_module("MaxPool2d", maxpool)
                else:
                    modules = maxpool

            elif module["type"] == "avgpool":
                kernel_size = module["size"]
                stride = module["stride"]
                modules.add_module("AvgPool2d", nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                                                             padding=(kernel_size - 1) // 2))

            elif module["type"] == "squeeze_excitation":
                in_channels = module["in_channels"]
                squeeze_channels = make_divisible(in_channels // 4, 8)
                modules.add_module("SeModule", SqueezeExcitation(in_channels,
                                                                 squeeze_channels,
                                                                 scale_activation=nn.Hardsigmoid))

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
                if self.onnx_export:  # explicitly state size, avoid scale_factor
                    g = (yolo_index + 1) * 2 / 32  # gain
                    modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
                else:
                    modules = nn.Upsample(scale_factor=module["stride"])

            elif module["type"] == "route":  # nn.Sequential() placeholder for "route" layer
                layers = module["layers"]
                filters = sum([output_filters[layer + 1 if layer > 0 else layer] for layer in layers])
                routs.extend([i + layer if layer < 0 else layer for layer in layers])
                modules = FeatureConcat(layers=layers)

            elif module["type"] == "shortcut":  # nn.Sequential() placeholder for "shortcut" layer
                layers = module["from"]
                filters = output_filters[-1]
                routs.extend([i + layer if layer < 0 else layer for layer in layers])
                modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in module)

            elif module["type"] == "reorg3d":  # yolov3_pytorch-spp-pan-scale
                pass

            elif module["type"] == "yolo":
                yolo_index += 1
                stride = [32, 16, 8]  # P5, P4, P3 strides
                if any(x in self.model_config_path for x in ["panet", "yolov4", "cd53"]):  # stride order reversed
                    stride = list(reversed(stride))
                layers = module["from"] if "from" in module else []
                modules = YOLOLayer(anchors=module["anchors"][module["mask"]],  # anchor list
                                    num_classes=module["classes"],  # number of classes
                                    img_size=img_size,  # (416, 416)
                                    yolo_index=yolo_index,  # 0, 1, 2...
                                    layers=layers,  # output layers
                                    stride=stride[yolo_index])

                # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
                try:
                    j = layers[yolo_index] if "from" in module else -1
                    # If previous layer is a dropout layer, get the one before
                    if self.module_defines[j].__class__.__name__ == "Dropout":
                        j -= 1
                    bias_ = self.module_defines[j][0].bias  # shape(255,)
                    bias = bias_[:modules.num_classes_output * modules.na].view(modules.na, -1)  # shape(3,85)
                    bias[:, 4] += -4.5  # obj
                    bias[:, 5:] += math.log(0.6 / (modules.num_classes - 0.99))  # cls (sigmoid(p) = 1/nc)
                    self.module_defines[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
                except:
                    pass

            elif module["type"] == "dropout":
                perc = float(module["probability"])
                modules = nn.Dropout(p=perc)
            else:
                print("Warning: Unrecognized Layer Type: " + module["type"])

            # Register module list and number of output filters
            module_list.append(modules)
            output_filters.append(filters)

        routs_binary = [False] * (i + 1)

        # Set YOLO route layer
        for i in routs:
            routs_binary[i] = True

        return module_list, routs_binary

    def get_yolo_layers(self):
        return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == "YOLOLayer"]

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout models
        print("Fusing layers...")
        fused_lists = nn.ModuleList()
        for layer in list(self.children())[0]:
            if isinstance(layer, nn.Sequential):
                for i, b in enumerate(layer):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = layer[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        layer = nn.Sequential(fused, *list(layer.children())[i + 1:])
                        break
            fused_lists.append(layer)
        self.module_list = fused_lists

    def forward(
            self,
            x: Tensor,
            augment: bool = False
    ) -> list[Any] | tuple[Tensor, Tensor] | tuple[Tensor, Any] | tuple[Tensor, None]:
        if not augment:
            return self.forward_once(x)
        else:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x, scale_img(x.flip(3), s[0], False), scale_img(x, s[1], False))):
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None

    def forward_once(
            self,
            x: Tensor,
            augment: bool = False) -> list[Any] | tuple[Tensor, Tensor] | tuple[Tensor, Any]:
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []

        # For augment
        batch_size = x.shape[0]
        scale_factor = [0.83, 0.67]

        # Augment imgs (inference and test only)
        if augment:
            x = torch.cat((x, scale_img(x.flip(3), scale_factor[0]), scale_img(x, scale_factor[1])), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name == "WeightedFeatureFusion":
                x = module(x, out)
            elif name == "FeatureConcat":
                x = module(out)
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:
                x = module(x)

            out.append(x if self.routs[i] else [])

        if self.training:  # train
            return yolo_out
        elif self.onnx_export:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, batch_size, dim=0)
                x[1][..., :4] /= scale_factor[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= scale_factor[1]  # scale
                x = torch.cat(x, 1)
            return x, p
