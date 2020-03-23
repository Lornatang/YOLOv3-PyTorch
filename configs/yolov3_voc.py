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
DATA = {
    "PATH": "/home/leon/data/data/VOC",
    "CLASSES": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'],
    "NUM_CLASSES": 20}

# model
YOLO = {"ANCHORS": [
    [(10, 13), (16, 30), (33, 23)],  # Anchors for small object
    [(30, 61), (62, 45), (59, 119)],  # Anchors for medium object
    [(116, 90), (156, 198), (373, 326)]  # Anchors for large object
],
    "STRIDES": [8, 16, 32],
    "MASK": 3
}

# train
TRAIN = {
    "BATCH_SIZE": 8,
    "WIDTH": 416,
    "HEIGHT": 416,
    "CHANNELS": 3,
    "MOMENTUM": 0.9,
    "LR": 0.001,
    "WEIGHT_DECAY": 0.0005,
    "DATA_AUGMENT": True,
    "MULTI_SCALE": True,
    "MAX_STEPS": 50200,  # 24.3 epoch
    "WARMUP_EPOCHS": 3,
    "IOU_THRESHOLD": 0.5,
}

# test
TEST = {
    "BATCH_SIZE": 1,
    "WIDTH": 416,
    "HEIGHT": 416,
    "CHANNELS": 3,
    "NUMBER_WORKERS": 4,
    "CONFIDENCE_THRESHOLD": 0.001,
    "NMS_THRESHOLD": 0.5,
    "DATA_AUGMENT": True,
    "MULTI_SCALE": False,
}
