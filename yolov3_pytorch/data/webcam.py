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

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F_vision

from .data_augment import letterbox

__all__ = [
    "LoadWebcam",
]


class LoadWebcam:
    def __init__(
            self,
            pipe: int = 0,
            img_size: int = 416,
            gray: bool = False,
    ) -> None:
        """Load images from a webcam.

        Args:
            pipe (int, optional): The webcam to use. Defaults: 0.
            img_size (int, optional): The size of the images. Defaults: 416.
            gray (bool, optional): Whether to convert the images to grayscale. Defaults: ``False``.
        """
        self.pipe = pipe
        self.img_size = img_size
        self.gray = gray

        if self.pipe == "0":
            self.pipe = 0  # local camera

        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        """Iterate over the images."""
        self.count = -1
        return self

    def __next__(self):
        """Get the next img."""
        self.count += 1
        if cv2.waitKey(1) == ord("q"):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, raw_img = self.cap.read()
        else:  # IP camera
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.count * 30)
            ret_val, raw_img = self.cap.read()

        assert ret_val, f"Camera Error {self.pipe}"

        if self.pipe == 0:  # local camera
            raw_img = cv2.flip(raw_img, 1)  # flip left-right

        # Print
        img_path = "webcam.jpg"
        print(f"webcam {self.count}: ", end="")

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

        return img_path, img, raw_img, None

    def __len__(self):
        """Number of images in the dataset."""
        return 0
