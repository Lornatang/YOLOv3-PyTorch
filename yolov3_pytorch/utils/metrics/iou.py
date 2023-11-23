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

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "box_iou", "bbox_iou", "wh_iou",
]


def box_iou(box1: Tensor or np.ndarray, box2: Tensor or np.ndarray) -> Tensor or np.ndarray:
    r"""Calculate the intersection-over-union (IoU) of boxes.

    Args:
        box1 (Tensor[N, 4]): Tensor containing N boxes in (x1, y1, x2, y2) format.
        box2 (Tensor[M, 4]): Tensor containing M boxes in (x1, y1, x2, y2) format.

    Returns:
        iou (Tensor[N, M]): Tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    def box_area(box):
        """
        Calculate the area of a box.

        Args:
            box (Tensor[4, n]): Tensor containing the coordinates of n boxes in (x1, y1, x2, y2) format.

        Returns:
            area (Tensor[n]): Tensor containing the area of each box.
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    # Calculate the areas of box1 and box2
    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # Calculate the intersection of box1 and box2
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    # Calculate the IoU
    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def bbox_iou(box1, box2, x1y1x2y2=True, g_iou=False, d_iou=False, c_iou=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # Calculate IoU

    if g_iou or d_iou or c_iou:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # Calculate convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # Calculate convex height

        if g_iou:
            c_area = cw * ch + 1e-16  # Calculate convex area
            return iou - (c_area - union) / c_area  # Calculate GIoU

        if d_iou or c_iou:
            c2 = cw ** 2 + ch ** 2 + 1e-16  # Calculate convex diagonal squared
            # Calculate centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4

            if d_iou:
                return iou - rho2 / c2

            elif c_iou:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    # Calculate alpha to mitigate the issue of non overlapping bounding boxes
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)

    return iou


def wh_iou(wh1, wh2):
    r"""Returns the IoU of two wh tensors

    Args:
        wh1 (Tensor): width and height of first tensor
        wh2 (Tensor): width and height of second tensor

    Returns:
        Tensor: IoU matrix of shape (N, M)
    """
    # Expand dimensions to create broadcasting shapes
    wh1 = wh1[:, None, :]  # [N, 1, 2]
    wh2 = wh2[None, :, :]  # [1, M, 2]

    # Calculate intersection and union areas
    inter = torch.min(wh1, wh2).prod(dim=2)  # [N, M]
    union = wh1.prod(dim=2) + wh2.prod(dim=2) - inter  # [N, M]

    # Calculate IoU
    iou = inter / union

    return iou
