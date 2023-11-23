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
Inference master program
"""
import argparse

from yolov3_pytorch.engine.inferencer import Inferencer


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch YOLOv3 Inference")
    parser.add_argument(
        "inputs",
        metavar="INPUTS",
        type=str,
        help="path to images or video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/inference/",
        help="path to outputs dir. Default: ``./results/inference``",
    )
    parser.add_argument(
        "--class-names-path",
        type=str,
        default="./data/voc.names",
        help="path to class names file. Default: ``./data/voc.names``",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="./model_configs/voc/yolov3_tiny.cfg",
        help="path to model config file. Default: ``./model_configs/voc/yolov3_tiny.cfg``",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="size of each image dimension. Default: 416",
    )
    parser.add_argument(
        "--gray",
        action="store_true",
        help="whether to grayscale image input",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="./results/pretrained_models/YOLOv3_Tiny-VOC0712-20231107.pth.tar",
        help="path to weights file. Default: ``./results/pretrained_models/YOLOv3_Tiny-VOC0712-20231107.pth.tar``",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="whether to half precision",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="whether to fuse conv and bn",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show image.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save results to *.txt.",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec (verify ffmpeg support). Default: ``mp4v``.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Object confidence threshold. Default: 0.25.")
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.45,
        help="IOU threshold for NMS. Default: 0.45.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Image augmented inference",
    )
    parser.add_argument(
        "--filter-classes",
        nargs="+",
        type=int,
        help="Filter by class",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Class-agnostic NMS",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device to use. Choice: ['cpu', 'gpu']. Default: ``gpu``",
    )
    opts = parser.parse_args()

    return opts


def main() -> None:
    opts = get_opts()

    app = Inferencer(opts)
    app.inference()


if __name__ == "__main__":
    main()
