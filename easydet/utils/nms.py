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
import torchvision

from .coords import xywh2xyxy
from .iou import bbox_iou
from .iou import box_iou


def non_max_suppression(prediction,
                        confidence_threshold=0.1,
                        iou_threshold=0.6,
                        multi_label=True,
                        classes=None,
                        agnostic=False):
    """
    Removes detections with lower object confidence score than "conf_thres"
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, conf, class)
    """

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    method = "vision_batch"
    batched = "batch" in method  # run once per image, all classes simultaneously
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Apply conf constraint
        pred = pred[pred[:, 4] > confidence_threshold]

        # Apply width-height constraint
        pred = pred[((pred[:, 2:4] > min_wh) & (pred[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Compute conf
        pred[..., 5:] *= pred[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(pred[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (pred[:, 5:] > confidence_threshold).nonzero().t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only
            conf, j = pred[:, 5:].max(1)
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes:
            pred = pred[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        if not torch.isfinite(pred).all():
            pred = pred[torch.isfinite(pred).all(1)]

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Sort by confidence
        if not method.startswith("vision"):
            pred = pred[pred[:, 4].argsort(descending=True)]

        # Batched NMS
        if batched:
            c = pred[:, 5] * 0 if agnostic else pred[:, 5]  # class-agnostic NMS
            boxes, scores = pred[:, :4].clone(), pred[:, 4]
            boxes += c.view(-1, 1) * max_wh
            if method == "vision_batch":
                i = torchvision.ops.boxes.nms(boxes, scores, iou_threshold)
            elif method == "fast_batch":  # FastNMS from https://github.com/dbolya/yolact
                iou = box_iou(boxes, boxes).triu_(diagonal=1)  # upper triangular iou matrix
                i = iou.max(dim=0)[0] < iou_threshold

            output[image_i] = pred[i]
            continue

        # All other NMS methods
        det_max = []
        cls = pred[:, -1]
        for c in cls.unique():
            dc = pred[cls == c]  # select class c
            n = len(dc)
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 500:
                dc = dc[:500]

            if method == "vision":
                det_max.append(dc[torchvision.ops.boxes.nms(dc[:, :4], dc[:, 4], iou_threshold)])

            elif method == "or":  # default
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we"re at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < iou_threshold]  # remove ious > threshold

            elif method == "and":  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < iou_threshold]  # remove ious > threshold

            elif method == "merge":  # weighted mixture box
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > iou_threshold  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

            elif method == "soft":  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    dc = dc[dc[:, 4] > iou_threshold]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output
