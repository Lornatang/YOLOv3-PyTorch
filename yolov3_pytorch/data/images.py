# Copyright 2023 Lornatang Authors. All Rights Reserved.
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
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision

from .data_augment import letterbox
from .utils import IMG_FORMATS, VID_FORMATS

__all__ = [
    "LoadImages",
]


class LoadImages:
    def __init__(
            self,
            img_path: Union[str, Path],
            img_size: int = 416,
            gray: bool = False,
    ) -> None:
        """Load images from a path.

        Args:
            img_path (str or Path): The path to the images
            img_size (int, optional): The size of the images. Defaults: 416
            gray (bool, optional): Whether to convert the images to grayscale. Defaults: ``False``
        """
        self.img_path = img_path
        self.img_size = img_size
        self.gray = gray

        self.files = []
        self.video_flag = []
        self.mode = "images"
        self.frame = 0
        self.cap = None
        self.num_frames = 0

        # Read image or video files
        if os.path.isdir(self.img_path):
            files = sorted(glob.glob(os.path.join(self.img_path, "*.*")))
        elif os.path.isfile(self.img_path):
            files = [self.img_path]
        else:
            raise TypeError(f"img_path must be a directory or path to an image, but got {type(self.img_path)}")

        # Check format of files
        for file in files:
            file_extension = file.split(".")[-1].lower()
            if file_extension in IMG_FORMATS:
                self.files.append(file)
                self.video_flag.append(False)
            elif file_extension in VID_FORMATS:
                self.files.append(file)
                self.video_flag.append(True)

        # number of files
        self.num_files = len(self.files)

        if any(self.video_flag):
            # new video
            self.new_video(self.files[0])
        else:
            self.cap = None

        assert self.num_files > 0, f"No images or videos found in {self.img_path}. " \
                                   f"Supported formats are:\n" \
                                   f"images: {IMG_FORMATS}\n" \
                                   f"videos: {VID_FORMATS}"

    def new_video(self, path: str) -> None:
        """Open a new video.

        Args:
            path (str): The path to the video.

        """
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_video(self, path):
        with cv2.VideoCapture(path) as cap:
            while True:
                ret_val, raw_img = cap.read()
                if not ret_val:
                    self.count += 1
                    if self.count == self.num_files:  # last video
                        raise StopIteration
                    else:
                        path = self.files[self.count]
                        self.new_video(path)
                        continue

                self.frame += 1
                print(f"video {self.count + 1}/{self.num_files} ({self.frame}/{self.num_frames}) {path}: ", end="")
                break

    def read_image(self, path) -> np.ndarray:
        self.count += 1
        raw_img = cv2.imread(path)
        assert raw_img is not None, "Image Not Found " + path
        print(f"image {self.count}/{self.num_files} {path}: ", end="")

        return raw_img

    def __iter__(self):
        """Iterate over the images."""
        self.count = 0
        return self

    def __next__(self):
        """Get the next image."""
        global raw_img
        if self.count == self.num_files:
            raise StopIteration

        path = self.files[self.count]
        video_flag = self.video_flag[self.count]

        if video_flag:
            # Read video
            self.mode = "video"
            self.read_video(path)
        else:
            # Read image
            raw_img = self.read_image(path)

        # Padded resize
        img = letterbox(raw_img, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # RGB numpy convert RGB tensor
        img = torch.from_numpy(img)

        if self.gray:
            # RGB tensor convert GRAY tensor
            img = F_vision.rgb_to_grayscale(img)

        return path, img, raw_img, self.cap

    def __len__(self):
        return self.num_files  # number of files
