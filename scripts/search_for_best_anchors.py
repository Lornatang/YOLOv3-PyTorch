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
import argparse

from yolov3_pytorch.utils.autochor import kmean_anchors


def main(opts):
    kmean_anchors(
        opts.path,
        opts.num_anchor,
        (opts.img_size, opts.img_size),
        opts.iou_thresh,
    )


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Search for best anchors in YOLOv3")
    parser.add_argument(
        "--path",
        type=str,
        default="./data/voc/train.txt",
        help="path to dataset. Default: ``./data/voc/train.txt``",
    )
    parser.add_argument(
        "--num-anchor",
        type=int,
        default=9,
        help="number of anchors. Default: 9",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="image size. Default: 416",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.25,
        help="iou threshold. Default: 0.25",
    )

    opts = parser.parse_args()

    return opts


if __name__ == "__main__":
    opts = get_opts()

    main(opts)
