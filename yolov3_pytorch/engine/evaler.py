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
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import boxes
from tqdm import tqdm

from yolov3_pytorch.data.datasets import LoadDatasets
from yolov3_pytorch.models.darknet import Darknet
from yolov3_pytorch.models.utils import load_state_dict
from yolov3_pytorch.utils.common import clip_coords, coco80_to_coco91_class, scale_coords, xywh2xyxy, xyxy2xywh
from yolov3_pytorch.utils.metrics import ap_per_class
from yolov3_pytorch.utils.nms import non_max_suppression

__all__ = [
    "Evaler",
]


class Evaler:
    def __init__(
            self,
            config: Dict,
            device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

    def load_datasets(self) -> DataLoader:
        r"""Load training and test datasets from a configuration file, such as yaml

        Returns:
            DataLoader: test_dataloader

        """
        if self.config["DATASET"]["SINGLE_CLASSES"] == 1:
            self.config["MODEL"]["NUM_CLASSES"] = 1
        self.config["TRAIN"]["LOSSES"]["CLS_LOSS"]["WEIGHT"] *= self.config["MODEL"]["NUM_CLASSES"] / 80

        test_datasets = LoadDatasets(self.config["DATASET"]["TEST_PATH"],
                                     self.config["TEST"]["IMG_SIZE"],
                                     self.config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                     False,
                                     self.config["AUGMENT"]["HYP"],
                                     self.config["DATASET"]["RECT_LABEL"],
                                     self.config["DATASET"]["CACHE_IMAGES"],
                                     self.config["DATASET"]["SINGLE_CLASSES"],
                                     pad=0.5,
                                     gray=self.config["MODEL"]["GRAY"])

        test_dataloader = DataLoader(test_datasets,
                                     batch_size=self.config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=False,
                                     persistent_workers=True,
                                     collate_fn=test_datasets.collate_fn)

        return test_dataloader

    def build_model(self) -> nn.Module:
        # Create model
        model = Darknet(self.config["MODEL"]["CONFIG_PATH"],
                        self.config["TRAIN"]["IMG_SIZE"],
                        self.config["MODEL"]["GRAY"],
                        self.config["MODEL"]["COMPILED"],
                        False)
        model = model.to(self.device)

        model.num_classes = self.config["MODEL"]["NUM_CLASSES"]

        # Compile model
        if self.config["MODEL"]["COMPILED"]:
            model = torch.compile(model)

        # Load model weights
        model_weights_path = self.config["TEST"]["WEIGHTS_PATH"]
        if model_weights_path.endswith(".pth.tar"):
            state_dict = torch.load(model_weights_path, map_location=self.device)["state_dict"]
            model = load_state_dict(model, state_dict)
        elif model_weights_path.endswith(".weights"):
            model.load_darknet_weights(model_weights_path)
        else:
            raise ValueError(f"'{model_weights_path}' is not supported.")
        print(f"Loaded `{model_weights_path}` models weights successfully.")

        return model

    def validate(self):
        test_dataloader = self.load_datasets()
        model = self.build_model()
        self.validate_on_epoch(
            model,
            test_dataloader,
            self.config["DATASET"]["CLASS_NAMES"],
            self.config["TEST"]["AUGMENT"],
            self.config["TEST"]["CONF_THRESH"],
            self.config["TEST"]["IOU_THRESH"],
            (self.config["TEST"]["IOUV1"], self.config["TEST"]["IOUV2"]),
            self.config["TEST"]["GT_JSON_PATH"],
            self.config["TEST"]["PRED_JSON_PATH"],
            self.config["TEST"]["VERBOSE"],
            self.device)

    @staticmethod
    def validate_on_epoch(
            model: nn.Module,
            dataloader: DataLoader,
            class_names: list,
            augment: bool,
            conf_thresh: float = 0.01,
            iou_thresh: float = 0.30,
            iouv: tuple = (0.5, 0.95),
            gt_json_path: str = "",
            pred_json_path: str = "",
            verbose: bool = False,
            device: torch.device = torch.device("cpu"),
    ) -> tuple:
        iouv = torch.linspace(iouv[0], iouv[1], 10).to(device)  # iou vector for mAP@0.5:0.95
        iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0

        # Put the models in eval mode
        model.eval()

        # if test coco91 dataset
        coco91class = coco80_to_coco91_class()

        # Format print information
        s = ("%20s" + "%10s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP@0.5", "F1")
        p, r, f1, mean_p, mean_r, mean_ap, mean_f1 = 0., 0., 0., 0., 0., 0., 0.
        jdict, stats, ap, ap_class = [], [], [], []

        for batch_index, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            batch_size, _, height, width = imgs.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)

            # Inference
            with torch.no_grad():
                output, _ = model(imgs, augment)  # inference and training outputs

            # Run NMS
            output = non_max_suppression(output, conf_thresh, iou_thresh)

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                num_labels = len(labels)
                target_classes = labels[:, 0].tolist() if num_labels else []  # target class
                seen += 1

                if pred is None:
                    if num_labels:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                      torch.Tensor(),
                                      torch.Tensor(),
                                      target_classes))
                    continue

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Append to pycocotools JSON dictionary
                if pred_json_path != "":
                    img_id = int(Path(paths[si]).stem.split("_")[-1])
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({"image_id": img_id,
                                      "category_id": coco91class[int(p[5])],
                                      "bbox": [round(x, 3) for x in b],
                                      "score": round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if num_labels:
                    detected = []  # target indices
                    target_classes_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(target_classes_tensor):
                        ti = (cls == target_classes_tensor).nonzero().view(-1)  # target indices
                        pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = boxes.box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            for j in (ious > iouv[0]).nonzero():
                                d = ti[i[j]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == num_labels:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, target_classes)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_classes))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            if niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mean_p, mean_r, mean_ap, mean_f1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            num_targets = np.bincount(stats[3].astype(np.int64), minlength=model.num_classes)  # number of targets per class
        else:
            num_targets = torch.zeros(1)

        # Print results
        pf = "%20s" + "%10d" + "%10d" + "%10.3g" * 4  # print format
        print(pf % ("all", seen, num_targets.sum(), mean_p, mean_r, mean_ap, mean_f1))

        # Print results per class
        if verbose and model.num_classes > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (class_names[c], seen, num_targets[c], p[i], r[i], ap[i], f1[i]))

        # Save JSON
        if pred_json_path != "" and mean_ap and len(jdict):
            print("\nCOCO mAP with pycocotools...")
            imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.image_files]
            with open(pred_json_path, "w") as file:
                json.dump(jdict, file)

            cocoGt = COCO(gt_json_path)
            cocoDt = cocoGt.loadRes(pred_json_path)  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        return mean_p, mean_r, mean_ap, mean_f1
