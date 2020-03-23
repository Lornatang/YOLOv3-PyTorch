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
    # Anchors for small object
    [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],
    # Anchors for medium object
    [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],
    # Anchors for large object
    [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],
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
