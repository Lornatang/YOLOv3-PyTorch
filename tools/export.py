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
import argparse
import os
from pathlib import Path

import torch

from yolov3_pytorch.models import Darknet
from yolov3_pytorch.models.utils import load_state_dict


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to onnx")
    parser.add_argument(
        "--weights",
        type=str,
        default="/path/to/weights",
        help="Model weights",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="model_configs/exp/yolov3.cfg",
        help="Model config file path",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="Image size",
    )
    parser.add_argument(
        "--gray",
        action="store_true",
        help="Gray mode",
    )
    parser.add_argument(
        "--export-mode",
        type=str,
        default="torch",
        choices=["torch"],
        help="Export mode",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default="results/export",
        help="Export path",
    )
    return parser.parse_args()


def main():
    opts = get_opts()

    os.makedirs(opts.export_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Darknet(opts.cfg, img_size=opts.img_size, gray=opts.gray)
    model = model.to(device)
    model.eval()

    if opts.weights.endswith(".pth.tar"):
        ckpt = torch.load(opts.weights, map_location=device)
        if "ema_state_dict" in ckpt:
            state_dict = ckpt["ema_state_dict"]
            model = load_state_dict(model, state_dict, False)
        else:
            raise RuntimeError("No 'ema_state_dict' in checkpoint file.")

    if opts.export_mode == "torch":
        out_weights = os.path.join(opts.export_dir, os.path.basename(opts.weights))
        torch.save({"state_dict": model.state_dict()}, out_weights)
        print(f"Exported model weights to {out_weights}")


if __name__ == "__main__":
    main()
