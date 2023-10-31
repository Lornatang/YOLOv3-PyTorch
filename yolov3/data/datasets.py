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
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


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


def load_image(self, index: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Loads an image from a file into a numpy array.

    Args:
        self: Dataset object
        index (int): Index of the image to load

    Returns:
        image (np.ndarray): Image as a numpy array

    """
    # loads 1 image from dataset, returns image, original hw, resized hw
    image = self.images[index]
    if image is None:  # not cached
        path = self.image_files[index]
        image = cv2.imread(path)  # BGR
        assert image is not None, "Image Not Found " + path
        h0, w0 = image.shape[:2]  # orig hw
        r = self.image_size / max(h0, w0)  # resize image to image_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.image_augment else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return image, (h0, w0), image.shape[:2]  # image, hw_original, hw_resized
    else:
        return self.images[index], self.image_hw0[index], self.image_hw[index]  # image, hw_original, hw_resized
