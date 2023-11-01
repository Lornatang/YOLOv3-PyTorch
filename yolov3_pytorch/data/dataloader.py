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
import glob
import os
import random
import time
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from torchvision.transforms import functional as F_vision
from tqdm import tqdm

from yolov3_pytorch.utils.common import xywh2xyxy, xyxy2xywh
from .data_augment import augment_hsv, load_mosaic, random_affine
from .datasets import letterbox, load_image

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

        # Cache images into memory for faster training (WARNING: large data may exceed system RAM)
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
