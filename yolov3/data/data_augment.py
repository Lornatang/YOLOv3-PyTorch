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
import random
from typing import Tuple, List

import cv2
import numpy as np
from numpy import ndarray

from .datasets import load_image


def augment_hsv(image: ndarray, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
    """Augment HSV channels of an image.

    Args:
        image (ndarray): Image to augment
        hgain (float): Hue gain
        sgain (float): Saturation gain
        vgain (float): Value gain

    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def cutout(image, labels):
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def load_mosaic(self, index: int) -> Tuple[np.ndarray, List]:
    """loads images in a mosaic

    Args:
        self: Dataset object
        index (int): Index of the image to load

    Returns:
        image (ndarray): Image as a numpy array

    """
    # loads images in a mosaic
    labels4 = []
    s = self.image_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        image, _, (h, w) = load_image(self, index)

        # place image in image4
        if i == 0:  # top left
            image4 = np.full((s * 2, s * 2, image.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # image4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    image4, labels4 = random_affine(image4,
                                    labels4,
                                    degrees=int(self.image_augment_dict["DEGREES"]),
                                    translate=float(self.image_augment_dict["TRANSLATE"]),
                                    scale=float(self.image_augment_dict["SCALE"]),
                                    shear=int(self.image_augment_dict["SHEAR"]),
                                    border=-s // 2)  # border to remove

    return image4, labels4


def random_affine(image, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    height = image.shape[0] + border * 2
    width = image.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(center=(image.shape[1] / 2, image.shape[0] / 2), angle=a, scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * image.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * image.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        image = cv2.warpAffine(image, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return image, targets