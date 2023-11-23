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
import numpy as np

__all__ = [
    "compute_ap", "ap_per_class",
]


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    r"""Compute the average precision, given the recall and precision curves.

    Args:
        recall (np.nparray): The recall curve.
        precision (np.nparray): The precision curve.

    Returns:
        float: The average precision as computed in py-faster-rcnn.
    """
    # Append sentinel values to beginning and end
    mean_recall = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mean_precision = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mean_precision = np.maximum.accumulate(mean_precision[::-1])[::-1]

    # Integrate area under curve
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mean_recall, mean_precision), x)  # integrate

    return ap


def ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray):
    r"""Computes the average precision, given the recall and precision curves.

    Args:
        tp (np.ndarray):True positives.
        conf (np.ndarray): Objectiveness value from 0-1.
        pred_cls (np.ndarray): Predicted object classes.
        target_cls (np.ndarray): True object classes.

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectiveness
    sorted_indices = np.argsort(-conf)
    tp, conf, pred_cls = tp[sorted_indices], conf[sorted_indices], pred_cls[sorted_indices]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    num_classes = unique_classes.shape[0]
    num_iou_thresh = tp.shape[1]
    ap, p, r = np.zeros((num_classes, num_iou_thresh)), np.zeros((num_classes, num_iou_thresh)), np.zeros((num_classes, num_iou_thresh))

    for ci, c in enumerate(unique_classes):
        is_class_c = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = is_class_c.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[is_class_c]).cumsum(0)
            tpc = tp[is_class_c].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[is_class_c], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[is_class_c], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(num_iou_thresh):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")
