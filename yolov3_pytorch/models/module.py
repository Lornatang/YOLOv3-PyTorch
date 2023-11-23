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
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch

__all__ = [
    "FeatureConcat", "InvertedResidual", "MixConv2d", "WeightedFeatureFusion", "YOLOLayer", "make_divisible", "fuse_conv_and_bn", "scale_img",
]


class FeatureConcat(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        r"""Initialize FeatureConcat module.

        Args:
            layers (nn.ModuleList): List of layers to concatenate.
        """
        super(FeatureConcat, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.multiple:
            x = torch.cat([x[i] for i in self.layers], dim=1)
        else:
            x = x[self.layers[0]]

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
            self.branch1 = nn.Identity()

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
            ch = np.linalg.solve(a, b).round().astype(int)  # solve for equal weight indices, ax = b

        mix_conv2d = [nn.Conv2d(in_channels=in_channels,
                                out_channels=ch[group],
                                kernel_size=kernel_size_tuple[group],
                                stride=stride,
                                padding=kernel_size_tuple[group] // 2,
                                dilation=dilation,
                                bias=bias) for group in range(groups)]
        self.mix_conv2d = nn.ModuleList(mix_conv2d)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([m(x) for m in self.mix_conv2d], dim=1)
        return x


class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers: nn.ModuleList, weight: bool = False) -> None:
        r"""Weighted Feature Fusion module.

        Args:
            layers: List of layer indices.
            weight: Flag to apply weights or not.
        """
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x: Tensor, outputs: Tensor) -> Tensor:
        r"""Forward pass of the Weighted Feature Fusion module.

        Args:
            x: Input tensor.
            outputs: List of output tensors from different layers.

        Returns:
            Tensor: Output tensor after feature fusion.
        """
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
            img_size: tuple,
            yolo_index: int,
            layers: list,
            stride: int,
            onnx_export: bool = False,
    ) -> None:
        """

        Args:
            anchors (list): List of anchors.
            num_classes (int): Number of classes.
            img_size (tuple): Image size.
            yolo_index (int): Yolo layer index.
            layers (list): List of layers.
            stride (int): Stride.
            onnx_export (bool, optional): Whether to export to onnx. Default: ``False``.

        """
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
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

    def create_grids(self, ng: tuple = (13, 13), device: str = "cpu"):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float, device=device)

        # Build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)],
                                    indexing="ij")
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        # Check if exporting to ONNX
        if self.onnx_export:
            bs = 1  # batch size
        else:
            # Get the shape of the input tensor
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13

            # Create grids if the shape has changed
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # Reshape and permute the tensor
        p = p.view(bs, self.na, self.num_classes_output, self.ny, self.nx)
        p = p.permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif self.onnx_export:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)  # Pre-calculate ng outside the loop
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


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    r"""Divisor to the number of channels.

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
    r"""Fuse convolution and batchnorm layers.

    Args:
        conv (nn.Conv2d): convolution layer
        bn (nn.BatchNorm2d): batchnorm layer

    Returns:
        fused_conv_bn (nn.Module): fused convolution layer
    """
    with torch.no_grad():
        # Initialize fused_conv_bn with the same parameters as conv
        fused_conv_bn = nn.Conv2d(conv.in_channels,
                                  conv.out_channels,
                                  kernel_size=conv.kernel_size,
                                  stride=conv.stride,
                                  padding=conv.padding,
                                  bias=True)

        # Fuse the convolution and batchnorm weights
        # Reshape the weight tensors for matrix multiplication
        w_conv = conv.weight.view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused_conv_bn.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv_bn.weight.size()))

        # Fuse the convolution and batchnorm biases
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused_conv_bn.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv_bn


def scale_img(img: Tensor, ratio: float = 1.0, same_shape: bool = True) -> Tensor:
    r"""Scales an image tensor by a ratio. If same_shape is True, the image is padded with zeros to maintain the same shape.

    Args:
        img (Tensor): Image tensor to be scaled
        ratio (float): Ratio to scale the image by
        same_shape (bool): Whether to pad the image with zeros to maintain the same shape

    Returns:
        Tensor: Scaled image tensor
    """
    # Scale the image
    height, width = img.shape[2:]
    new_size: Tuple[int, int] = (int(height * ratio), int(width * ratio))
    img = F_torch.interpolate(img, size=new_size, mode="bilinear", align_corners=False)

    if not same_shape:
        # Pad or crop the image
        grid_size = 64  # (pixels) grid size
        height, width = [math.ceil(x * ratio / grid_size) * grid_size for x in (height, width)]

    img = F_torch.pad(img, [0, width - new_size[1], 0, height - new_size[0]], value=0.447)

    return img
