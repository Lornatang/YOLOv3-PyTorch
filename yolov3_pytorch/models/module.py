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
from typing import Any, List, Dict, Optional, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision.ops.misc import SqueezeExcitation


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
