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
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Mapping

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision.ops.misc import SqueezeExcitation


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

        self.module_defines = self.parse_model_config_path()
        self.module_list, self.routs = self.create_modules()
        self.yolo_layers = self.get_yolo_layers()

        # compile keyword
        if compile_mode:
            compile_state = "_orig_mod"

        # Darknet parameters
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def parse_model_config_path(self) -> List[Dict[str, Any]]:
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

    def create_modules(self) -> [nn.ModuleList, list]:
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
                    modules.add_module("MixConv2d", _MixConv2d(in_channels=output_filters[-1],
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
                squeeze_channels = _make_divisible(in_channels // 4, 8)
                modules.add_module("SeModule", SqueezeExcitation(in_channels,
                                                                 squeeze_channels,
                                                                 scale_activation=nn.Hardsigmoid))

            elif module["type"] == "InvertedResidual":
                in_channels = module["in_channels"]
                out_channels = module["out_channels"]
                stride = module["stride"]
                modules.add_module("InvertedResidual", _InvertedResidual(in_channels=in_channels,
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
                modules = _FeatureConcat(layers=layers)

            elif module["type"] == "shortcut":  # nn.Sequential() placeholder for "shortcut" layer
                layers = module["from"]
                filters = output_filters[-1]
                routs.extend([i + layer if layer < 0 else layer for layer in layers])
                modules = _WeightedFeatureFusion(layers=layers, weight="weights_type" in module)

            elif module["type"] == "reorg3d":  # yolov3_pytorch-spp-pan-scale
                pass

            elif module["type"] == "yolo":
                yolo_index += 1
                stride = [32, 16, 8]  # P5, P4, P3 strides
                if any(x in self.model_config_path for x in ["panet", "yolov4", "cd53"]):  # stride order reversed
                    stride = list(reversed(stride))
                layers = module["from"] if "from" in module else []
                modules = _YOLOLayer(anchors=module["anchors"][module["mask"]],  # anchor list
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
        return [i for i, m in enumerate(self.module_defines) if m.__class__.__name__ == "_YOLOLayer"]

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
                        fused = _fuse_conv_and_bn(conv, b)
                        layer = nn.Sequential(fused, *list(layer.children())[i + 1:])
                        break
            fused_lists.append(layer)
        self.module_list = fused_lists

    def load_torch_state_dict(
            self,
            state_dict: Mapping[str, Any],
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
        """

        # When the PyTorch version is less than 2.0, the model compilation is not supported.
        if int(torch.__version__[0]) < 2 and self.compile_mode:
            warnings.warn("PyTorch version is less than 2.0, does not support model compilation.")
            self.compile_mode = False

        # Create new OrderedDict that does not contain the module prefix
        model_state_dict = self.module_list.state_dict()
        new_state_dict = OrderedDict()

        # Remove the module prefix and update the model weight
        for k, v in state_dict.items():
            current_compile_state = k.split(".")[0]

            if current_compile_state == self.compile_state and not self.compile_mode:
                name = k[len(self.compile_state) + 1:]
            elif current_compile_state != self.compile_state and self.compile_mode:
                raise ValueError("The model is not compiled, but the weight is compiled.")
            else:
                name = k
            new_state_dict[name] = v
        state_dict = new_state_dict

        # Filter out unnecessary parameters
        new_state_dict = {k: v for k, v in state_dict.items() if
                          k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

        # Update model parameters
        model_state_dict.update(new_state_dict)
        self.module_list.load_state_dict(model_state_dict)

    def load_darknet_weights(self, weights_path: Union[str, Path]) -> None:
        r"""Parses and loads the weights stored in 'weights_path'
        
        Args:
            weights_path (Union[str, Path]): Path to the weights file.
        """

        # Open the weights file
        with open(weights_path, "rb") as f:
            # First five are header values
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of imgs seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        # If the weights file has a cutoff, we can find out about it by looking at the filename
        # examples: darknet53.conv.74 -> cutoff is 74
        filename = os.path.basename(weights_path)
        if ".conv." in filename:
            try:
                cutoff = int(filename.split(".")[-1])  # use last part of filename
            except ValueError:
                pass

        ptr = 0
        for i, (module_define, module) in enumerate(zip(self.module_defines, self.module_list)):
            if i == cutoff:
                break
            if module_define["type"] == "convolutional":
                conv_layer = module[0]
                if module_define["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, weights_path: Union[str, Path], cutoff: int = -1) -> None:
        r"""Saves the models in darknet format.
        
        Args:
            weights_path: path of the new weights file
            cutoff: save layers between 0 and cutoff (cutoff = -1 -> all are saved) Default: -1
        """

        fp = open(weights_path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_define, module) in enumerate(zip(self.module_defines[:cutoff], self.module_list[:cutoff])):
            if module_define["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_define["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def forward(
            self,
            x: Tensor,
            img_augment: bool = False
    ) -> list[Any] | tuple[Tensor, Tensor] | tuple[Tensor, Any] | tuple[Tensor, None]:
        if not img_augment:
            return self.forward_once(x)
        else:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x, _scale_img(x.flip(3), s[0], False), _scale_img(x, s[1], False))):
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
            x = torch.cat((x, _scale_img(x.flip(3), scale_factor[0]), _scale_img(x, scale_factor[1])), 0)

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name == "_WeightedFeatureFusion":
                x = module(x, out)
            elif name == "_FeatureConcat":
                x = module(out)
            elif name == "_YOLOLayer":
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


class _YOLOLayer(nn.Module):
    def __init__(
            self,
            anchors: list,
            num_classes: int,
            img_size: tuple,
            yolo_index: int,
            layers: list,
            stride: int,
            onnx_export: bool = False,
    ) -> None:
        r"""

        Args:
            anchors (list): List of anchors.
            num_classes (int): Number of classes.
            img_size (tuple): Image size.
            yolo_index (int): Yolo layer index.
            layers (list): List of layers.
            stride (int): Stride.
            onnx_export (bool, optional): Whether to export to onnx. Default: ``False``.
        """

        super(_YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # models output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.num_classes = num_classes  # number of classes (80)
        self.num_classes_output = num_classes + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y grid points
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.onnx_export = onnx_export
        self.grid = None

        if onnx_export:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)],
                                    indexing="ij")
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if self.onnx_export:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.num_classes_output, self.ny, self.nx)
        p = p.permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif self.onnx_export:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.num_classes_output)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.num_classes == 1 else \
                torch.sigmoid(p[:, 5:self.num_classes_output]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.num_classes_output), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class _FeatureConcat(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """

        Args:
            layers (nn.ModuleList):

        """
        super(_FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([x[i] for i in self.layers], 1) if self.multiple else x[self.layers[0]]

        return x


class _MixConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size_tuple: tuple = (3, 5, 7),
            stride: int = 1,
            dilation: int = 1,
            bias: bool = True,
            method: str = "equal_params") -> None:
        """MixConv: Mixed Depth-Wise Convolutional Kernels https://arxiv.org/abs/1907.09595

        Args:
            in_channels (int): Number of channels in the input img
            out_channels (int): Number of channels produced by the convolution
            kernel_size_tuple (tuple, optional): A tuple of 3 different kernel sizes. Defaults to (3, 5, 7).
            stride (int, optional): Stride of the convolution. Defaults to 1.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            method (str, optional): Method to split channels. Defaults to "equal_params".

        """
        super(_MixConv2d, self).__init__()

        groups = len(kernel_size_tuple)

        if method == "equal_ch":  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_channels).floor()  # out_channels indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # "equal_params": equal parameter count per group
            b = [out_channels] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size_tuple) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        mix_conv2d = []
        for group in range(groups):
            mix_conv2d.append(nn.Conv2d(in_channels=in_channels,
                                        out_channels=ch[group],
                                        kernel_size=kernel_size_tuple[group],
                                        stride=stride,
                                        padding=kernel_size_tuple[group] // 2,
                                        dilation=dilation,
                                        bias=bias))
        self.mix_conv2d = nn.ModuleList(*mix_conv2d)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([m(x) for m in self.mix_conv2d], dim=1)

        return x


class _WeightedFeatureFusion(nn.Module):
    def __init__(self, layers: nn.ModuleList, weight: bool = False) -> None:
        """

        Args:
            layers:
            weight:

        """
        super(_WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x: Tensor, outputs: Tensor) -> Tensor:
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class _InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(_InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("Illegal stride value")
        self.stride = stride

        branch_features = out_channels // 2
        assert (self.stride != 1) or (in_channels == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depth_wise_conv(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depth_wise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride,
                                 padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depth_wise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = F_torch.channel_shuffle(out, 2)

        return out


def _fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Module:
    """Fuse convolution and batchnorm layers.

    Args:
        conv (nn.Conv2d): convolution layer
        bn (nn.BatchNorm2d): batchnorm layer

    Returns:
        fused_conv_bn (nn.Module): fused convolution layer

    """
    with torch.no_grad():
        # init
        fused_conv_bn = nn.Conv2d(conv.in_channels,
                                  conv.out_channels,
                                  kernel_size=conv.kernel_size,
                                  stride=conv.stride,
                                  padding=conv.padding,
                                  bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused_conv_bn.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv_bn.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused_conv_bn.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fused_conv_bn


def _scale_img(img: Tensor, ratio: float = 1.0, same_shape: bool = True) -> Tensor:
    """Scales an img by a ratio. If same_shape is True, the img is padded with zeros to maintain the same shape.

    Args:
        img (Tensor): img to be scaled
        ratio (float): ratio to scale img by
        same_shape (bool): whether to pad img with zeros to maintain same shape

    Returns:
        img (Tensor): scaled img

    """
    # scales img(bs,3,y,x) by ratio
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F_torch.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 64  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]

    img = F_torch.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

    return img


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Divisor to the number of channels.

    Args:
        v (float): input value
        divisor (int): divisor
        min_value (int): minimum value

    Returns:
        int: divisible value
    """

    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v
