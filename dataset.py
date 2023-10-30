# Copyright 2022 Lorna Authors. All Rights Reserved.
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
import glob
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from typing import Any, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from numpy import ndarray
from scipy.cluster.vq import kmeans
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F_vision
from tqdm import tqdm

__all__ = [
    "parse_dataset_config", "load_image", "augment_hsv", "load_mosaic", "letterbox", "random_affine", "cutout",
    "xywh2xyxy", "xyxy2xywh", "labels_to_class_weights",
    "LoadImages", "LoadStreams", "LoadWebcam",
    "LoadImagesAndLabels"
]

support_image_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
support_video_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def _exif_size(image: Image.Image) -> tuple:
    """Get the size of an image from its EXIF data.

    Args:
        image (Image.Image): The image to get the size from.

    Returns:
        image_size (tuple): The size of the image.

    """
    # Returns exif-corrected PIL size
    image_size = image.size  # (width, height)
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            image_size = (image_size[1], image_size[0])
        elif rotation == 8:  # rotation 90
            image_size = (image_size[1], image_size[0])
    except:
        pass

    return image_size


def parse_dataset_config(path: str) -> dict:
    """Parses the data configuration file

    Args:
        path (str): path to data config file

    Returns:
        data_config (dict): A dictionary containing the information from the data config file
    
    """
    if not os.path.exists(path) and os.path.exists("data" + os.sep + path):  # add data/ prefix if omitted
        path = "data" + os.sep + path

    with open(path, "r") as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        key, val = line.split("=")
        options[key.strip()] = val.strip()

    return options


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


def kmean_anchors(
        path: str = "./data/VOC0712/train.txt",
        num_anchor: int = 9,
        image_size: tuple = (608, 608),
        iou_threshold: float = 0.50,
        gen: int = 1000):
    """Compute kmean anchors for dataset

    Args:
        path (str): path to dataset
        num_anchor (int): number of anchors
        image_size (tuple): image size
        iou_threshold (float): iou threshold
        gen (int): number of generation

    Returns:
        nparray: kmean anchors
    """

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        for i, x in enumerate(k):
            print(f"{round(x[0])},{round(x[1])}", end=",  " if i < len(k) - 1 else "\n")  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > iou_threshold).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, image_augment=True, rect_label=True)
    nr = 1 if image_size[0] == image_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(image_size[0], image_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    print(f"Running kmeans for {num_anchor} anchors on {len(wh)} points...")
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, num_anchor, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # Evolve
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc="Evolving anchors"):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((np.random.random(sh) < mp) * np.random.random() * np.random.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k


def wh_iou(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Returns the IoU of two set of boxes, wh1 is 1st set of bboxes, wh2 is 2nd set of bboxes

    Args:
        wh1 (Tensor): tensor of bounding boxes, Shape: [nb_target, 2]
        wh2 (Tensor): tensor of bounding boxes, Shape: [nb_bboxes, 2]

    Returns:
        Tensor: IoU, Shape: [nb_target, nb_bboxes]

    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


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


class LoadImages:  # for inference
    def __init__(self, images_path: str, image_size: int = 416, gray: bool = False) -> None:
        """Load images from a path.

        Args:
            images_path (str): The path to the images.
            image_size (int, optional): The size of the images. Defaults: 416.
            gray (bool, optional): Whether to convert the images to grayscale. Defaults: ``False``.

        """
        images_path = str(Path(images_path))  # os-agnostic
        files = []

        if os.path.isdir(images_path):
            files = sorted(glob.glob(os.path.join(images_path, "*.*")))
        elif os.path.isfile(images_path):
            files = [images_path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in support_image_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in support_video_formats]
        nI, nV = len(images), len(videos)

        self.image_size = image_size
        self.gray = gray
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = "images"
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, f"No images or videos found in {images_path}. " \
                            f"Supported formats are:\n" \
                            f"images: {support_image_formats}\n" \
                            f"videos: {support_video_formats}"

    def __iter__(self):
        """Iterate over the images."""
        self.count = 0
        return self

    def __next__(self):
        """Get the next image."""
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, raw_image = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, raw_image = self.cap.read()

            self.frame += 1
            print(f"video {self.count + 1}/{self.nF} ({self.frame}/{self.nframes}) {path}: ", end="")

        else:
            # Read image
            self.count += 1
            raw_image = cv2.imread(path)  # BGR
            assert raw_image is not None, "Image Not Found " + path
            print(f"image {self.count}/{self.nF} {path}: ", end="")

        # Padded resize
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        # RGB numpy convert RGB tensor
        image = torch.from_numpy(image)

        if self.gray:
            # RGB tensor convert GRAY tensor
            image = F_vision.rgb_to_grayscale(image)

        return path, image, raw_image, self.cap

    def new_video(self, path: str) -> None:
        """Open a new video.

        Args:
            path (str): The path to the video.

        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe: int = 0, image_size: int = 416, gray: bool = False) -> None:
        """Load images from a webcam.

        Args:
            pipe (int, optional): The webcam to use. Defaults: 0.
            image_size (int, optional): The size of the images. Defaults: 416.
            gray (bool, optional): Whether to convert the images to grayscale. Defaults: ``False``.

        """
        self.image_size = image_size
        self.gray = gray

        if pipe == "0":
            pipe = 0  # local camera
        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        """Iterate over the images."""
        self.count = -1
        return self

    def __next__(self):
        """Get the next image."""
        self.count += 1
        if cv2.waitKey(1) == ord("q"):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, raw_image = self.cap.read()
            raw_image = cv2.flip(raw_image, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, raw_image = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f"Camera Error {self.pipe}"
        image_path = "webcam.jpg"
        print(f"webcam {self.count}: ", end="")

        # Padded resize
        image = letterbox(raw_image, new_shape=self.image_size)[0]

        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        # RGB numpy convert RGB tensor
        image = torch.from_numpy(image)

        if self.gray:
            # RGB tensor convert GRAY tensor
            image = F_vision.rgb_to_grayscale(image)

        return image_path, image, raw_image, None

    def __len__(self):
        """Number of images in the dataset."""
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources="streams.txt", image_size=416, gray: bool = False) -> None:
        """Load multiple IP or RTSP cameras.

        Args:
            sources (str, optional): The path to the file with the sources. Defaults: "streams.txt".
            image_size (int, optional): The size of the images. Defaults: 416.

        """
        self.mode = "images"
        self.image_size = image_size
        self.gray = gray

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.images = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            cap = cv2.VideoCapture(0 if s == "0" else s)
            assert cap.isOpened(), "Failed to open %s" % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.images[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {fps:.2f} FPS).")
            thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.image_size)[0].shape for x in self.images], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print("WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.")

    def update(self, index, cap):
        """Update a single stream."""
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.images[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.images[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        """Iterate over the images."""
        self.count = -1
        return self

    def __next__(self):
        """Get the next image."""
        self.count += 1
        raw_image = self.images.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        image = [letterbox(x, new_shape=self.image_size, auto=self.rect)[0] for x in raw_image]

        # Stack
        image = np.stack(image, 0)

        # Convert BGR to RGB
        image = image[:, :, :, ::-1].transpose(0, 3, 1, 2)
        image = np.ascontiguousarray(image)

        # RGB numpy convert RGB tensor
        image = torch.from_numpy(image)

        if self.gray:
            # RGB tensor convert GRAY tensor
            image = F_vision.rgb_to_grayscale(image)

        return self.sources, image, raw_image, None

    def __len__(self):
        """Number of images in the dataset."""
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImagesAndLabels(Dataset):
    def __init__(
            self,
            path: str,
            image_size: int = 416,
            batch_size: int = 16,
            image_augment: bool = False,
            image_augment_dict: Any = None,
            rect_label: bool = False,
            image_weights: bool = False,
            cache_images: bool = False,
            single_classes: bool = False,
            pad: float = 0.0,
            gray: bool = False,
    ) -> None:
        """Load images and labels.

        Args:
            path (str): The path to the images.
            image_size (int, optional): The size of the images. Defaults: 416.
            batch_size (int, optional): The size of the batch. Defaults: 16.
            image_augment (bool, optional): Whether to augment the images. Defaults: ``False``.
            image_augment_dict (Any, optional): The image augment configure. Defaults: None.
            rect_label (bool, optional): Whether to use rectangular trainning. Defaults: ``False``.
            image_weights (bool, optional): Whether to use image weights. Defaults: ``False``.
            cache_images (bool, optional): Whether to cache the images. Defaults: ``False``.
            single_classes (bool, optional): Whether to use single class. Defaults: ``False``.
            pad (float, optional): The padding. Defaults: 0.0.
            gray (bool, optional): Whether to use grayscale. Defaults: ``False``.

        """
        try:
            path = str(Path(path))  # os-agnostic
            parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):  # file
                with open(path, "r") as f:
                    f = f.read().splitlines()
                    f = [x.replace("./", parent) if x.startswith("./") else x for x in f]  # local to global path
            elif os.path.isdir(path):  # folder
                f = glob.iglob(path + os.sep + "*.*")
            else:
                raise Exception(f"{path} does not exist")
            self.image_files = [x.replace("/", os.sep) for x in f if
                                os.path.splitext(x)[-1].lower() in support_image_formats]
        except:
            raise Exception(f"Error loading data from {path}")

        num_images = len(self.image_files)
        assert num_images > 0, f"No images found in {path}"
        batch_index = np.floor(np.arange(num_images) / batch_size).astype(np.int16)  # batch index
        nb = batch_index[-1] + 1  # number of batches

        self.num_images = num_images  # number of images
        self.batch_index = batch_index  # batch index of image
        self.image_size = image_size
        self.image_augment = image_augment
        self.image_augment_dict = image_augment_dict
        self.image_weights = image_weights
        self.rect_label = False if image_weights else rect_label
        self.mosaic = self.image_augment and not self.rect_label  # load 4 images at a time into a mosaic (only during training)
        self.gray = gray

        # Define labels
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.image_files]

        # Read image shapes (wh)
        sp = path.replace(".txt", "") + ".shapes"  # shapefile path
        try:
            with open(sp, "r") as f:  # read existing shapefile
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == num_images, "Shapefile out of sync"
        except:
            s = [_exif_size(Image.open(f)) for f in tqdm(self.image_files, desc="Reading image shapes")]

        self.shapes = np.asarray(s, dtype=np.float64)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect_label:
            # Sort by aspect ratio
            s = self.shapes  # wh
            aspect_ratio = s[:, 1] / s[:, 0]  # aspect ratio
            index_rect = aspect_ratio.argsort()
            self.image_files = [self.image_files[i] for i in index_rect]
            self.label_files = [self.label_files[i] for i in index_rect]
            self.shapes = s[index_rect]  # wh
            aspect_ratio = aspect_ratio[index_rect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = aspect_ratio[batch_index == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * image_size / 32. + pad).astype(np.int16) * 32

        # Cache labels
        self.images = [None] * num_images
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * num_images
        create_data_subset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        s = path.replace("images", "labels")

        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            if labels_loaded:
                labels = self.labels[i]
            else:
                try:
                    with open(file, "r") as f:
                        labels = np.asarray([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except:
                    nm += 1
                    continue

            if labels.shape[0]:
                assert labels.shape[1] == 5, f"> 5 label columns: {file}"
                assert (labels >= 0).all(), f"negative labels: {file}"
                assert (labels[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinate labels: {file}"
                if np.unique(labels, axis=0).shape[0] < labels.shape[0]:  # duplicate rows
                    nd += 1
                if single_classes:
                    labels[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = labels
                nf += 1  # file found

                # Create sub dataset (a smaller dataset)
                if create_data_subset and ns < 1E4:
                    if ns == 0:
                        os.makedirs(os.path.join("samples", "data_subset", "images"), exist_ok=True)
                    exclude_classes = 43
                    if exclude_classes not in labels[:, 0]:
                        ns += 1
                        with open(os.path.join("data_subset", "images.txt"), "a") as f:
                            f.write(self.image_files[i] + "\n")

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.image_files[i])
                    image = cv2.imread(str(p))
                    h, w = image.shape[:2]
                    for j, x in enumerate(labels):
                        f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent, exist_ok=True)  # make new output folder

                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int16)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, image[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:
                ne += 1  # file empty

            pbar.desc = f"Caching labels {s} ({nf} found, {nm} missing, {ne} empty, {nd} duplicate, for {num_images} images)"
        assert nf > 0 or num_images == 20288, f"No labels found in {os.path.dirname(file) + os.sep}."

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.image_files)), desc="Caching images")
            self.image_hw0, self.image_hw = [None] * num_images, [None] * num_images
            for i in pbar:  # max 10k images
                self.images[i], self.image_hw0[i], self.image_hw[i] = load_image(self, i)
                gb += self.images[i].nbytes
                pbar.desc = f"Caching images ({gb / 1e9:.1f}GB)"

        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.image_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except:
                    print(f"Corrupted image detected: {file}")

    def __len__(self):
        """Number of images."""
        return len(self.image_files)

    def __getitem__(self, index: int):
        """Returns the image and label at the specified index."""
        if self.image_weights:
            index = self.indices[index]

        if self.mosaic:
            # Load mosaic
            image, labels = load_mosaic(self, index)
            shapes = None

        else:
            # Load image
            image, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[
                self.batch_index[index]] if self.rect_label else self.image_size  # final letterboxed shape
            image, ratio, pad = letterbox(image, shape, auto=False, scaleup=self.image_augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.image_augment:
            # Augment image space
            if not self.mosaic:
                image, labels = random_affine(image, labels,
                                              degrees=self.image_augment_dict["DEGREES"],
                                              translate=self.image_augment_dict["TRANSLATE"],
                                              scale=self.image_augment_dict["SCALE"],
                                              shear=self.image_augment_dict["SHEAR"])

            # Augment colorspace
            augment_hsv(image,
                        hgain=self.image_augment_dict["HSV_H"],
                        sgain=self.image_augment_dict["HSV_S"],
                        vgain=self.image_augment_dict["HSV_V"])

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= image.shape[0]  # height
            labels[:, [1, 3]] /= image.shape[1]  # width

        if self.image_augment:
            # random left-right flip
            if self.image_augment_dict["USE_LR_FLIP"] and random.random() < 0.5:
                image = np.fliplr(image)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            if self.image_augment_dict["USE_UD_FLIP"] and random.random() < 0.5:
                image = np.flipud(image)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        # RGB numpy convert RGB tensor
        image = torch.from_numpy(image)

        if self.gray:
            # RGB tensor convert GRAY tensor
            image = F_vision.rgb_to_grayscale(image)

        return image, labels_out, self.image_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        image, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(image, 0), torch.cat(label, 0), path, shapes
