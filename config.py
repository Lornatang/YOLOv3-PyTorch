# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.deterministic = False
cudnn.benchmark = True
# Model arch name
model_arch_name = "yolov3_tiny_voc"
# Set to True if the label is rectangular
train_rect_label = True
test_rect_label = True
# Set to True if there is only 1 detection classes
single_classes = False
# If use grayscale image
gray = False
# Export ONNX model
onnx_export = False
# For test
conf_threshold = 0.001
iou_threshold = 0.5
save_json = False
train_augment = True
test_augment = False
verbose = False
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "YOLOv3_tiny-VOC0712"

hyper_parameters_dict = {
    "giou": 3.54,  # giou loss gain
    "cls": 37.4,  # cls loss gain
    "cls_pw": 1.0,  # cls BCELoss positive_weight
    "obj": 64.3,  # obj loss gain
    "obj_pw": 1.0,  # obj BCELoss positive_weight
    "iou_t": 0.20,  # iou training threshold
    "fl_gamma": 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
    "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
    "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
    "degrees": 1.98 * 0,  # image rotation (+/- deg)
    "translate": 0.05 * 0,  # image translation (+/- fraction)
    "scale": 0.05 * 0,  # image scale (+/- gain)
    "shear": 0.641 * 0  # image shear (+/- gain)
}

if mode == "train":
    # Dataset config for training
    dataset_config_path = f"./data/voc.data"

    # Default use multi-scale training
    train_image_size_min = 416
    train_image_size_max = 608
    test_image_size = 416
    grid_size = 32  # Do not modify

    batch_size = 64
    num_workers = 4

    # Load the address of the pre-trained model
    pretrained_model_weights_path = f"./results/pretrained_models/YOLOv3_tiny-COCO.weights"

    # Define this parameter when training is interrupted or migrated
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 300

    # Optimizer parameter
    optim_lr = 1e-2
    optim_momentum = 0.937
    optim_weight_decay = 5e-4

    # EMA moving average model parameters
    model_ema_decay = 0.999

    # How many iterations to print the training result
    train_print_frequency = 200
    test_print_frequency = 1

if mode == "test":
    # Dataset config for test
    test_dataset_config_path = f"./data/voc.data"

    test_image_size = 608
    batch_size = 64
    num_workers = 4

    model_weights_path = ""
