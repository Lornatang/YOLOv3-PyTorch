# Copyright 2020 Lorna Authors. All Rights Reserved.
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
import torch

import torch.nn as nn
from cfgs import yolov3_voc
from ..layer import YOLO
from ..backbone import Tiny


class TinyVOC(nn.Module):
    def __init__(self):
        super(TinyVOC, self).__init__()

        self.num_anchors = torch.Tensor(yolov3_voc.YOLO["ANCHORS"])
        self.strides = torch.Tensor(yolov3_voc.YOLO["STRIDES"])
        self.num_classes = yolov3_voc.DATA["NUM_CLASSES"]
        self.out_channels = yolov3_voc.YOLO["MASK"] * (
                self.num_classes + 5)

        self.backbone = Tiny()

        # small anchors
        self.small = YOLO(anchors=self.num_anchors[0],
                          num_classes=self.num_classes,
                          stride=self.strides[0])
        # medium anchors
        self.medium = YOLO(anchors=self.num_anchors[1],
                           num_classes=self.num_classes,
                           stride=self.strides[1])

    def forward(self, x):
        out = []

        small, medium = self.backnone(x)

        out.append(self.small(small))
        out.append(self.medium(medium))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # small, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
