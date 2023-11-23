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
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .common import xywh2xyxy

__all__ = [
    "plot_one_box", "plot_images",
]


def plot_one_box(
        xyxy: tuple,
        img: np.ndarray,
        color: list[int] or tuple[int] = None,
        label: str = None,
        line_thickness: float = None
) -> None:
    """Plots one bounding box on the image.

    Args:
        xyxy (tuple): Bounding box coordinates (x1, y1, x2, y2).
        img (np.ndarray): Image to plot on.
        color (list[int] | tuple[int]): Color of the box (RGB values).
        label (str): Label of the box.
        line_thickness (float): Thickness of the lines of the box.

    """
    # Calculate line thickness based on image size
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

    # Generate random color if not provided
    color = color or [random.randint(0, 255) for _ in range(3)]

    # Convert bounding box coordinates to integers
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

    # Draw the bounding box rectangle on the image
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        # Calculate font thickness
        tf = max(tl - 1, 1)

        # Calculate text size
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        # Calculate the coordinates for the label background rectangle
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # Draw the label background rectangle
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)

        # Draw the label text
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_images(
        imgs: Tensor,
        targets: Tensor,
        paths: str = None,
        file_name: str = "images.jpg",
        names: str = None,
        max_size: int = 640,
        max_subplots: int = 16,
) -> None:
    """Plots images with bounding boxes

    Args:
        imgs (Tensor): images to plot
        targets (Tensor): targets to plot
        paths (str): paths to images
        file_name (str): name of the file to save
        names (str): names of the classes
        max_size (int): maximum size of the image
        max_subplots (int): maximum number of subplots

    """
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if os.path.isfile(file_name):
        return None

    imgs = imgs.cpu().numpy() if isinstance(imgs, torch.Tensor) else imgs
    targets = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    imgs *= 255  # un-normalize

    bs, _, h, w = imgs.shape[:4]  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = int(np.ceil(bs ** 0.5))  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h, w = math.ceil(scale_factor * h), math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((ns * h, ns * w, 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    color_lut = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in prop_cycle.by_key()["color"]]

    for i, img in enumerate(imgs):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")
            gt = image_targets.shape[1] == 6  # ground truth if no conf column
            conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic,
                        label,
                        (block_x + 5, block_y + t_size[1] + 5),
                        0,
                        tl / 3,
                        [220, 220, 220],
                        thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if file_name is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(file_name, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic
