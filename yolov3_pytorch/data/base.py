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
"""
Base class for datasets.
"""
import glob
import os
import random
from pathlib import Path
from typing import Any, Tuple, List

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import ExifTags, Image
from skimage import io
from torchvision.transforms import functional as F_vision
from tqdm import tqdm

from yolov3_pytorch.utils.common import xywh2xyxy, xyxy2xywh
from .data_augment import adjust_hsv, letterbox, random_affine
from .utils import IMG_FORMATS

__all__ = [
    "BaseDatasets",
]

# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


class BaseDatasets(torch.utils.data.Dataset):
    def __init__(
            self,
            path: str,
            img_size: int = 416,
            batch_size: int = 16,
            augment: bool = False,
            augment_dict: Any = None,
            rect_label: bool = False,
            img_weights: bool = False,
            cache_imgs: bool = False,
            single_classes: bool = False,
            pad: float = 0.0,
            gray: bool = False,
    ) -> None:
        """Load images and labels.

        Args:
            path (str): The path to the images.
            img_size (int, optional): The size of the images. Defaults: 416.
            batch_size (int, optional): The size of the batch. Defaults: 16.
            augment (bool, optional): Whether to augment the images. Defaults: ``False``.
            augment_dict (Any, optional): The image augment configure. Defaults: None.
            rect_label (bool, optional): Whether to use rectangular training. Defaults: ``False``.
            img_weights (bool, optional): Whether to use image weights. Defaults: ``False``.
            cache_imgs (bool, optional): Whether to cache the images. Defaults: ``False``.
            single_classes (bool, optional): Whether to use single class. Defaults: ``False``.
            pad (float, optional): The padding. Defaults: 0.0.
            gray (bool, optional): Whether to use grayscale. Defaults: ``False``.

        """
        try:
            # Convert the path to a string and handle cross-platform issues
            path = str(Path(path))
            parent = str(Path(path).parent) + os.sep

            if os.path.isfile(path):
                with open(path, "r") as f:
                    # Read the file contents and split by lines
                    lines = f.read().splitlines()
                    # Replace paths starting with "./" with the parent directory path
                    lines = [x.replace("./", parent) if x.startswith("./") else x for x in lines]

            elif os.path.isdir(path):
                # Use the glob module to get all files in the directory
                lines = glob.iglob(path + os.sep + "*.*")

            else:
                # If it's neither a file nor a directory, raise an exception
                raise Exception(f"{path} does not exist")

            # Add file paths that match the specified image formats to the self.image_files list
            self.img_files = [x.replace("/", os.sep) for x in lines if x.split(".")[-1].lower() in IMG_FORMATS]

        except:
            # If an exception occurs, raise an exception with an error message
            raise Exception(f"Error occurred while loading data from {path}")

        # Calculate the number of images
        num_imgs = len(self.img_files)
        assert num_imgs > 0, f"No images found in {path}"

        # Calculate the batch index and the number of batches
        batch_idx = np.floor(np.arange(num_imgs) / batch_size).astype(np.int16)  # batch index
        nb = batch_idx[-1] + 1  # number of batches

        # Set the attributes of the data loader
        self.num_imgs = num_imgs
        self.batch_idx = batch_idx
        self.img_size = img_size
        self.augment = augment
        self.augment_dict = augment_dict
        self.img_weights = img_weights
        self.rect_label = False if img_weights else rect_label
        self.mosaic = self.augment and not self.rect_label  # load 4 images at a time into a mosaic (only during training)
        self.gray = gray
        self.indices = None

        # Define labels
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt") for x in self.img_files]

        # Read image shapes (width and height)
        shapefile_path = path.replace(".txt", "") + ".shapes"
        try:
            # read existing shapefile
            with open(shapefile_path, "r") as f:
                s = [x.split() for x in f.read().splitlines()]
                assert len(s) == num_imgs, "Shapefile out of sync"
        except:
            # If the shapefile doesn't exist, calculate the image shapes using the _exif_size function
            s = [self._exif_size(Image.open(f)) for f in tqdm(self.img_files, desc="Reading image shapes")]

        # Convert the image shapes to a NumPy array
        self.shapes = np.asarray(s, dtype=np.float64)

        if self.rect_label:
            # Sort images and labels by aspect ratio
            s = self.shapes  # wh
            aspect_ratio = s[:, 1] / s[:, 0]  # aspect ratio
            index_rect = aspect_ratio.argsort()
            self.img_files = [self.img_files[i] for i in index_rect]
            self.label_files = [self.label_files[i] for i in index_rect]
            self.shapes = s[index_rect]  # wh
            aspect_ratio = aspect_ratio[index_rect]

            # Set training image shapes based on aspect ratio
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = aspect_ratio[batch_idx == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            # Calculate batch shapes for training
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int16) * 32

        # Cache labels
        self.imgs = [None] * num_imgs
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * num_imgs
        create_data_subset, extract_bounding_boxes, labels_loaded = False, False, False
        num_missing, num_found, num_empty, num_subset, num_duplicate = 0, 0, 0, 0, 0
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
                    num_missing += 1
                    continue

            if labels.shape[0]:
                assert labels.shape[1] == 5, f"> 5 label columns: {file}"
                assert (labels >= 0).all(), f"negative labels: {file}"
                assert (labels[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinate labels: {file}"
                if np.unique(labels, axis=0).shape[0] < labels.shape[0]:  # duplicate rows
                    num_duplicate += 1
                if single_classes:
                    labels[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = labels
                num_found += 1

                # Create sub dataset (a smaller dataset)
                if create_data_subset and num_subset < 1E4:
                    if num_subset == 0:
                        os.makedirs(os.path.join("samples", "data_subset", "images"), exist_ok=True)
                    exclude_classes = 43
                    if exclude_classes not in labels[:, 0]:
                        num_subset += 1
                        with open(os.path.join("data_subset", "images.txt"), "a") as f:
                            f.write(self.img_files[i] + "\n")

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
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
                num_empty += 1  # file empty

            pbar.desc = (f"Caching labels {s} "
                         f"({num_found} found, "
                         f"{num_missing} missing, "
                         f"{num_empty} empty, "
                         f"{num_duplicate} duplicate, "
                         f"for {num_imgs} images)")
        assert num_found > 0 or num_imgs == 20288, f"No labels found in {os.path.dirname(file) + os.sep}."

        # Cache images into memory for faster training (WARNING: large data may exceed system RAM)
        if cache_imgs:  # if training
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            self.image_hw0, self.image_hw = [None] * num_imgs, [None] * num_imgs
            for i in pbar:  # max 10k images
                self.imgs[i], self.image_hw0[i], self.image_hw[i] = self.load_image(i)
                gb += self.imgs[i].nbytes
                pbar.desc = f"Caching images ({gb / 1e9:.1f}GB)"

        # Detect corrupted images
        detect_corrupted_images = False
        if detect_corrupted_images:
            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except:
                    print(f"Corrupted image detected: {file}")

    @staticmethod
    def _exif_size(img: Image.Image) -> tuple:
        """Get the size of an image from its EXIF data.

        Args:
            img (Image.Image): The image to get the size from.

        Returns:
            image_size (tuple): The size of the image.
        """

        # Returns exif-corrected PIL size
        img_size = img.size  # (width, height)
        try:
            rotation = dict(img._getexif().items())[orientation]
            if rotation == 6:  # rotation 270
                img_size = (img_size[1], img_size[0])
            elif rotation == 8:  # rotation 90
                img_size = (img_size[1], img_size[0])
        except:
            pass

        return img_size

    def load_image(
            self,
            index: int,
    ) -> tuple[np.ndarray | np.ndarray[Any, np.dtype[np.generic | np.generic]] | Any, tuple[int, int], tuple[int, ...]] | tuple[None, None, None]:
        """
        Loads an image from a file into a numpy array.

        Args:
            self (Dataset): Dataset object
            index (int): Index of the image to load

        Returns:
            tuple: A tuple containing the loaded image, original height and width, and resized height and width.
        """
        img = self.imgs[index]  # Get the image from the cache
        if img is None:  # If the image is not cached
            path = self.img_files[index]  # Get the file path of the image
            img = cv2.imread(path)  # Read the image using OpenCV (BGR format)
            assert img is not None, "Image Not Found " + path  # Check if the image was successfully loaded
            h0, w0 = img.shape[:2]  # Get the original height and width of the image
            r = self.img_size / max(h0, w0)  # Calculate the resize ratio
            if r != 1:  # If the image needs to be resized
                # Choose the interpolation method based on the resize ratio and augmentation flag
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]
        else:
            return self.imgs[index], self.image_hw0[index], self.image_hw[index]

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
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place image in image4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
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

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # image4[ymin:ymax, xmin:xmax]
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
        img4, labels4 = random_affine(img4,
                                      labels4,
                                      degrees=int(self.augment_dict["DEGREES"]),
                                      translate=float(self.augment_dict["TRANSLATE"]),
                                      scale=float(self.augment_dict["SCALE"]),
                                      shear=int(self.augment_dict["SHEAR"]),
                                      border=-s // 2)  # border to remove

        return img4, labels4

    def __len__(self):
        """Number of images."""
        return len(self.img_files)

    def __getitem__(self, index: int):
        """Returns the image and label at the specified index."""
        if self.img_weights:
            index = self.indices[index]

        if self.mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch_idx[index]] if self.rect_label else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
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

        if self.augment:
            # Augment image space
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=self.augment_dict["DEGREES"],
                                            translate=self.augment_dict["TRANSLATE"],
                                            scale=self.augment_dict["SCALE"],
                                            shear=self.augment_dict["SHEAR"])

            # Augment colorspace
            img = adjust_hsv(img,
                             h_gain=self.augment_dict["HSV_H"],
                             s_gain=self.augment_dict["HSV_S"],
                             v_gain=self.augment_dict["HSV_V"])

        num_labels = len(labels)
        if num_labels:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            if self.augment_dict["USE_LR_FLIP"] and random.random() < 0.5:
                img = np.fliplr(img)
                if num_labels:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            if self.augment_dict["USE_UD_FLIP"] and random.random() < 0.5:
                img = np.flipud(img)
                if num_labels:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((num_labels, 6))
        if num_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # RGB numpy convert RGB tensor
        img = torch.from_numpy(img)

        if self.gray:
            # RGB tensor convert GRAY tensor
            img = F_vision.rgb_to_grayscale(img)

        return img, labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        # transposed
        img, label, path, shapes = zip(*batch)
        for i, data in enumerate(label):
            data[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
