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
Training master program.
All training scripts are scheduled by this script
"""
import argparse
import os
import yaml
import torch
import numpy as np
import random
from torch.backends import cudnn
from torch.cuda import amp

from yolov3_pytorch.engine.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch YOLOv3 Training")
    parser.add_argument("config_path", metavar="FILE", help="path to config file")
    opts = parser.parse_args()

    return opts


def init(config) -> tuple:
    # Fixed random number seed
    seed = config["SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Create a folder to save the model and log
    save_weights_dir = os.path.join("samples", config["EXP_NAME"])
    save_logs_dir = os.path.join("samples", "logs", config["EXP_NAME"])
    os.makedirs(save_weights_dir, exist_ok=True)
    os.makedirs(save_logs_dir, exist_ok=True)

    tblogger = SummaryWriter(save_logs_dir)

    return save_weights_dir, save_logs_dir, tblogger, scaler, device


def main() -> None:
    opts = get_opts()

    # Read YAML configuration file
    with open(opts.config_path, "r") as f:
        config = yaml.full_load(f)

    save_weights_dir, save_logs_dir, tblogger, scaler, device = init(config)

    app = Trainer(config, save_weights_dir, save_logs_dir, tblogger, scaler, device)
    app.train()


if __name__ == "__main__":
    main()
