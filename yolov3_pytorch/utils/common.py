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
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def clip_coords(boxes: Tensor, image_shape: tuple) -> Tensor:
    """Clip bounding xyxy bounding boxes to image shape (height, width)

    Args:
        boxes (Tensor): xyxy bounding boxes, shape (n, 4)
        image_shape (tuple): (height, width)
    """
    boxes[:, 0].clamp_(0, image_shape[1])  # x1
    boxes[:, 1].clamp_(0, image_shape[0])  # y1
    boxes[:, 2].clamp_(0, image_shape[1])  # x2
    boxes[:, 3].clamp_(0, image_shape[0])  # y2


def coco80_to_coco91_class() -> list:
    """Converts COCO80 class indices to COCO91 class indices.

    Returns:
        list: COCO91 class indices.

    """

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def labels_to_class_weights(labels: Tensor, num_classes: int = 80) -> Tensor:
    """Compute the class weights for the dataset.

    Args:
        labels (Tensor): A tensor of shape (N, ) where N is the number of labels.
        num_classes (int, optional): The number of classes. Defaults to 80.

    Returns:
        Tensor: A tensor of shape (num_classes, ) containing the class weights.

    """
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int16)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=num_classes)  # occurences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    weights = torch.from_numpy(weights)

    return weights


def letterbox(
        image: ndarray,
        new_shape: int or tuple = (416, 416),
        color: tuple = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scaleup: bool = True
) -> tuple[Any, tuple[float | Any, float | Any], tuple[float | int | Any, float | int | Any]]:
    """Resize image to a 32-pixel-multiple rectangle.

    Args:
        image (ndarray): Image to resize
        new_shape (int or tuple): Desired output shape of the image
        color (tuple): Color of the border
        auto (bool): Whether to choose the smaller dimension as the new shape
        scale_fill (bool): Whether to stretch the image to fill the new shape
        scaleup (bool): Whether to scale up the image if the image is smaller than the new shape

    Returns:
        ndarray: Resized image

    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return image, ratio, (dw, dh)


def parse_dataset_config(config_path: Union[str, Path]) -> dict:
    r"""Parses the data configuration file

    Args:
        config_path (str or Path): path to data config file

    Returns:
        data_config (dict): A dictionary containing the information from the data config file
    """

    # Open the config file and read all lines
    with open(config_path, "r") as config_file:
        lines = config_file.readlines()

    # Dictionary to store the config options
    config_options = {}
    for line in lines:
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Skip empty lines and comment lines
        if line == "" or line.startswith("#"):
            continue
        # Split the line into key and value based on the "=" delimiter
        key, value = line.split("=")
        # Remove whitespace from the key and value, and store in the dictionary
        config_options[key.strip()] = value.strip()

    return config_options


def scale_coords(new_image_shape, coords, raw_image_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        # gain  = old / new
        gain = max(new_image_shape) / max(raw_image_shape)
        # wh padding
        pad = (new_image_shape[1] - raw_image_shape[1] * gain) / 2, \
              (new_image_shape[0] - raw_image_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, raw_image_shape)
    return coords


def xywh2xyxy(x: ndarray) -> ndarray:
    """Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2]

    Args:
        x (ndarray): bounding boxes, sized [N,4].

    Returns:
        ndarray: converted bounding boxes, sized [N,4].
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x: ndarray) -> ndarray:
    """Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h]

    Args:
        x (ndarray): bounding boxes, sized [N,4].

    Returns:
        ndarray: converted bounding boxes, sized [N,4].

    """
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
