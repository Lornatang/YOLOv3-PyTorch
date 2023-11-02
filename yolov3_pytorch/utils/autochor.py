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
import torch
from scipy.cluster.vq import kmeans
from torch import Tensor
from tqdm import tqdm


def wh_iou(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Returns the IoU of two set of boxes, wh1 is 1st set of bboxes, wh2 is 2nd set of bboxes

    Args:
        wh1 (Tensor): tensor of bounding boxes, Shape: [nb_target, 2]
        wh2 (Tensor): tensor of bounding boxes, Shape: [nb_bboxes, 2]

    Returns:
        Tensor: IoU, Shape: [nb_target, nb_bboxes]

    """
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def kmean_anchors(
        path: str = "./data/voc/train.txt",
        num_anchor: int = 9,
        image_size: tuple = (416, 416),
        iou_threshold: float = 0.50,
        gen: int = 1000):
    """Compute kmean anchors for dataset

    Args:
        path (str): path to dataset
        num_anchor (int): number of anchors
        image_size (tuple): image size
        iou_threshold (float): iou threshold
        gen (int): number of generation

    Returns:
        nparray: kmean anchors
    """

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        for i, x in enumerate(k):
            print(f"{round(x[0])},{round(x[1])}", end=",  " if i < len(k) - 1 else "\n")  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > iou_threshold).float()).mean()  # product

    # Get label wh
    wh = []
    from yolov3_pytorch.data import LoadImagesAndLabels
    dataset = LoadImagesAndLabels(path, image_augment=True, rect_label=True)
    nr = 1 if image_size[0] == image_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(image_size[0], image_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    print(f"Running kmeans for {num_anchor} anchors on {len(wh)} points...")
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, num_anchor, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # Evolve
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc="Evolving anchors"):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((np.random.random(sh) < mp) * np.random.random() * np.random.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k
