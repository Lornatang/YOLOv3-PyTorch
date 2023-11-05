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
from typing import Any, Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch


class FeatureConcat(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """

        Args:
            layers (nn.ModuleList):

        """
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([x[i] for i in self.layers], 1) if self.multiple else x[self.layers[0]]

        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(InvertedResidual, self).__init__()

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


class MixConv2d(nn.Module):
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
        super(MixConv2d, self).__init__()

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


class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers: nn.ModuleList, weight: bool = False) -> None:
        """

        Args:
            layers:
            weight:

        """
        super(WeightedFeatureFusion, self).__init__()
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


class YOLOLayer(nn.Module):
    def __init__(
            self,
            anchors: list,
            num_classes: int,
            image_size: tuple,
            yolo_index: int,
            layers: list,
            stride: int,
            onnx_export: bool = False,
    ) -> None:
        """

        Args:
            anchors (list): List of anchors.
            num_classes (int): Number of classes.
            image_size (tuple): Image size.
            yolo_index (int): Yolo layer index.
            layers (list): List of layers.
            stride (int): Stride.
            onnx_export (bool, optional): Whether to export to onnx. Default: ``False``.

        """
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index
        self.layers = layers
        self.stride = stride
        self.num_layers = len(layers)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_classes_output = num_classes + 5
        self.num_x, self.num_y, self.num_gird = 0, 0, 0
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2)
        self.onnx_export = onnx_export
        self.grid = None

        if onnx_export:
            self.training = False
            self.create_grids((image_size[1] // stride, image_size[0] // stride))  # number x, y grid points

    def create_grids(self, num_grid=(13, 13), device="cpu"):
        self.num_x, self.num_y = num_grid  # x and y grid size
        self.num_gird = torch.tensor(num_grid, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.num_y, device=device),
                                     torch.arange(self.num_x, device=device)],
                                    indexing="ij")
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.num_y, self.num_x, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, x: Tensor) -> tuple[Tensor | Any, float | Any, float | Any] | tuple[Any, Any] | Any:
        if self.onnx_export:
            batch_size = 1  # batch size
        else:
            batch_size, _, ny, nx = x.shape  # bs, 255, 13, 13
            if (self.num_x, self.num_y) != (nx, ny):
                self.create_grids((nx, ny), x.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        x = x.view(batch_size, self.num_anchors, self.num_classes_output, self.num_y, self.num_x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return x

        elif self.onnx_export:
            # Avoid broadcasting for ANE operations
            m = self.num_anchors * self.num_x * self.num_y
            num_grid = 1. / self.num_gird.repeat(m, 1)
            grid = self.grid.repeat(1, self.num_anchors, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.num_x, self.num_y, 1).view(m, 2) * num_grid

            x = x.view(m, self.num_classes_output)
            xy = torch.sigmoid(x[:, 0:2]) + grid  # x, y
            wh = torch.exp(x[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(x[:, 4:5]) if self.num_classes == 1 else \
                torch.sigmoid(x[:, 5:self.num_classes_output]) * torch.sigmoid(x[:, 4:5])  # conf
            return p_cls, xy * num_grid, wh
        # inference
        else:
            inference_x = x.clone()
            inference_x[..., :2] = torch.sigmoid(inference_x[..., :2]) + self.grid  # xy
            inference_x[..., 2:4] = torch.exp(inference_x[..., 2:4]) * self.anchor_wh  # wh yolo method
            inference_x[..., :4] *= self.stride
            inference_x = torch.sigmoid_(inference_x[..., 4:])
            # view [1, 3, 13, 13, 85] as [1, 507, 85]
            inference_x = inference_x.view(batch_size, -1, self.num_classes_output)

            return inference_x, x


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
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


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Module:
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


def scale_img(img: Tensor, ratio: float = 1.0, same_shape: bool = True) -> Tensor:
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
