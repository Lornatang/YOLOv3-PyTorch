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
import sys
sys.path.append("..")
import torch.nn as nn
from model import YOLO
from model import Darknet19

DATA_PATH = "/home/unix/dataset/VOC"
PROJECT_PATH = "/home/unix/code/One-Stage-Detector/yolo"

DATA = {"CLASSES": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor'],
        "NUM": 80}

# model
MODEL = {"ANCHORS": [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],
                     # Anchors for small obj
                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],
                     # Anchors for medium obj
                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],
         # Anchors for big obj
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3
         }

# train
TRAIN = {
    "TRAIN_IMG_SIZE": 448,
    "AUGMENT": True,
    "BATCH_SIZE": 8,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2  # or None
}

# test
TEST = {
    "TEST_IMG_SIZE": 544,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": False,
    "FLIP_TEST": False
}


class Yolov3(nn.Module):
    def __init__(self):
        super(Yolov3, self).__init__()

        self.num_anchors = torch.Tensor(MODEL["ANCHORS"])
        self.strides = torch.Tensor(MODEL["STRIDES"])
        self.num_classes = DATA["NUM"]
        self.out_channels = MODEL["ANCHORS_PER_SCLAE"] * (self.num_classes + 5)

        self.backnone = Darknet19()

        # small
        self.small = YOLO(anchors=self.num_anchors[0],
                          num_classes=self.num_classes,
                          stride=self.strides[0])
        # medium
        self.medium = YOLO(anchors=self.num_anchors[1],
                           num_classes=self.num_classes,
                           stride=self.strides[1])

    def forward(self, x):
        out = []

        pred_small, pred_medium = self.backnone(x)

        out.append(self.small(pred_small))
        out.append(self.medium(pred_medium))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium,
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)
