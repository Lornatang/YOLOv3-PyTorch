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
import time

import torch
import torchvision.ops
from torch import Tensor

from .common import xywh2xyxy
from .metrics.iou import box_iou

__all__ = [
    "non_max_suppression",
]


def non_max_suppression(prediction: Tensor,
                        conf_thresh: float = 0.1,
                        iou_thresh: float = 0.6,
                        multi_label: bool = True,
                        filter_classes: list = None,
                        agnostic: bool = False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    # merge for best mAP
    merge = True
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    # seconds to quit after
    timeout = 3.0

    start_time = time.time()

    # number of classes
    num_classes = prediction[0].shape[1] - 5
    # multiple labels per box
    multi_label &= num_classes > 1
    output = [None] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):
        # Apply constraints
        x = x[x[:, 4] > conf_thresh]  # confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thresh).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thresh]

        # Filter by class
        if filter_classes:
            x = x[(j.view(-1, 1) == torch.tensor(filter_classes, device=j.device)).any(1)]

        # If none remain process next image
        # number of boxes
        num_boxes = x.shape[0]
        if not num_boxes:
            continue

        # Batched NMS
        classes = x[:, 5] * 0 if agnostic else x[:, 5]
        boxes, scores = x[:, :4].clone() + classes.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thresh)
        # Merge NMS (boxes merged using weighted mean)
        if merge and (1 < num_boxes < 3E3):
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thresh  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            except:
                print(x, i, x.shape, i.shape)
                pass

        output[img_idx] = x[i]
        if (time.time() - start_time) > timeout:
            break

    return output
