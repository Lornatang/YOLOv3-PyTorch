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
import os
import time
from threading import Thread

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision

from .data_augment import letterbox

__all__ = [
    "LoadStreams",
]


class LoadStreams:
    def __init__(
            self,
            sources="streams.txt",
            img_size=416,
            gray: bool = False,
    ) -> None:
        """Load multiple IP or RTSP cameras.

        Args:
            sources (str, optional): The path to the file with the sources. Defaults: "streams.txt".
            img_size (int, optional): The size of the images. Defaults: 416.
        """
        self.mode = "images"
        self.img_size = img_size
        self.gray = gray

        # Read the sources from a file or use a single source
        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources

        # Iterate through the sources and start reading frames from the video stream
        for i, s in enumerate(sources):
            print(f"{i + 1}/{n}: {s}... ", end="")
            cap = cv2.VideoCapture(0 if s == "0" else s)
            assert cap.isOpened(), "Failed to open %s" % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {fps:.2f} FPS).")
            thread.start()
        print("")  # newline

        # Check if all the streams have the same shape
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print("WARNING: Different stream shapes detected. For optimal performance, supply similarly-shaped streams.")

    def update(self, index, cap):
        """Update a single stream."""
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        """Iterate over the images."""
        return self

    def __next__(self):
        """Get the next image."""
        self.count += 1
        raw_img = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img = np.concatenate([letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in raw_img], axis=0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = torch.as_tensor(img)

        if self.gray:
            img = F_vision.rgb_to_grayscale(img)

        return self.sources, img, raw_img, None

    def __len__(self):
        """Number of images in the dataset."""
        return len(self.imgs)
