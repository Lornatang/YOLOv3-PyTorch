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
from typing import Any

import torch
from torch import nn, Tensor

from yolov3_pytorch.utils.metrics import bbox_iou, wh_iou


class BCEBlurWithLogitsLoss(nn.Module):
    r"""BCEwithLogitLoss() with reduced missing label effects."""

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        # must be nn.BCEWithLogitsLoss()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # prob from logits
        pred = torch.sigmoid(pred)
        # reduce only missing label effects
        dx = pred - true

        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        loss = loss.mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn: nn.Module, gamma: float = 1.5, alpha: float = 0.25):
        """Focal loss for binary classification

        Args:
            loss_fcn (nn.Module): loss function
            gamma (float, optional): gamma. Defaults to 1.5.
            alpha (float, optional): alpha. Defaults to 0.25.
        """
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        # required to apply FL to each element
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        # prob from logits
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def build_targets(
        p: Tensor,
        targets: Tensor,
        model: nn.Module,
        iou_thresh: float = 0.5,
) -> tuple[list[Any], list[Tensor], list[tuple[Any, Tensor | list[Any] | Any, Any, Any]], list[Any]]:
    """Build targets for compute_loss(), input targets(img,class,x,y,w,h)

    Args:
        iou_thresh:
        p (Tensor): predictions
        targets (Tensor): targets
        model (nn.Module): models

    Returns:
        tuple[list[Any], list[Tensor], list[tuple[Any, Tensor | list[Any] | Any, Any, Any]], list[Any]]: targets, indices, anchors, regression
    """
    # Build targets for compute_loss(), input targets(img,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    for i, j in enumerate(model.yolo_layers):
        anchors = model.module_list[j].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]].to(targets.device)  # xyxy gain
        na = anchors.shape[0]  # number of anchors
        # anchor tensor, same as .repeat_interleave(nt)
        at = torch.arange(na).view(na, 1).repeat(1, nt).to(targets.device)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            j = wh_iou(anchors, t[:, 4:6]) > iou_thresh
            # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        # Define
        b, c = t[:, :2].long().T  # img, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append# img, anchor, grid indices
        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets
            assert c.max() < model.num_classes, f"Model accepts {model.num_classes} classes labeled from 0-{model.num_classes - 1}, however you labelled a class {c.max()}. "
    return tcls, tbox, indices, anch


def compute_loss(
        p: Tensor,
        targets: Tensor,
        model: nn.Module,
        iou_thresh: float,
        losses_dict: Any,
):  # predictions, targets, models
    """Computes loss for YOLOv3.

    Args:
        p (Tensor): predictions
        targets (Tensor): targets
        model (nn.Module): models
        iou_thresh (float): iou threshold
        losses_dict (Any): losses dict

    Returns:
        loss (Tensor): loss

    """
    lcls = torch.FloatTensor([0]).to(device=targets.device)
    lbox = torch.FloatTensor([0]).to(device=targets.device)
    lobj = torch.FloatTensor([0]).to(device=targets.device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model, iou_thresh)  # targets

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([losses_dict["CLS_BCE_PW_LOSS"]["WEIGHT"]]).to(targets.device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([losses_dict["OBJ_BCE_PW_LOSS"]["WEIGHT"]]).to(targets.device))

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_bce(eps=0.0)

    # focal loss
    g = losses_dict["FL_GAMMA_LOSS"]["WEIGHT"]  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # img, anchor, gridy, gridx
        target_obj = torch.zeros_like(pi[..., 0])  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = ps[:, :2].sigmoid()
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, g_iou=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Obj
            target_obj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(target_obj.dtype)  # giou ratio

            # Class
            if model.num_classes > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        lobj += BCEobj(pi[..., 4], target_obj)  # obj loss

    lbox *= losses_dict["GIOU_LOSS"]["WEIGHT"]
    lobj *= losses_dict["OBJ_LOSS"]["WEIGHT"]
    lcls *= losses_dict["CLS_LOSS"]["WEIGHT"]

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


def smooth_bce(eps: float = 0.1):
    r"""Label smoothing epsilon

    Args:
        eps (float, optional): epsilon. Default: 0.1

    Returns:
        tuple[float, float]: positive, negative label smoothing BCE targets
    """

    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
