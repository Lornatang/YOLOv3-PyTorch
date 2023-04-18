# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.ops import boxes
from tqdm import tqdm
from typing import Any
import model
import test_config
from dataset import parse_dataset_config, LoadImagesAndLabels
from utils import load_classes, load_pretrained_torch_state_dict, load_pretrained_darknet_state_dict, ap_per_class, \
    clip_coords, coco80_to_coco91_class, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh


def main():
    device = torch.device(test_config.device)
    # Fixed random number seed
    random.seed(test_config.seed)
    np.random.seed(test_config.seed)
    torch.manual_seed(test_config.seed)
    torch.cuda.manual_seed_all(test_config.seed)

    test_dataloader, num_classes, names = build_dataset()
    yolo_model = build_model(num_classes)

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    test(yolo_model,
         test_dataloader,
         names,
         iouv,
         niou,
         test_config,
         device)


def build_dataset() -> [nn.Module, int, list]:
    # Load dataset
    dataset_dict = parse_dataset_config(test_config.test_dataset_config_path)
    num_classes = 1 if test_config.single_classes else int(dataset_dict["classes"])
    names = load_classes(dataset_dict["names"])

    test_datasets = LoadImagesAndLabels(path=dataset_dict["test"],
                                        image_size=test_config.test_image_size,
                                        batch_size=test_config.batch_size,
                                        image_augment=test_config.test_augment,
                                        rect_label=test_config.test_rect_label,
                                        cache_images=test_config.cache_images,
                                        single_classes=test_config.single_classes,
                                        pad=0.5,
                                        gray=test_config.gray)
    # generate dataset iterator
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=test_config.batch_size,
                                 shuffle=False,
                                 num_workers=test_config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True,
                                 collate_fn=test_datasets.collate_fn)

    return test_dataloader, num_classes, names


def build_model(num_classes: int) -> nn.Module:
    # Create model
    yolo_model = model.__dict__[test_config.model_arch_name](image_size=(test_config.test_image_size,
                                                                         test_config.test_image_size),
                                                             gray=test_config.gray,
                                                             onnx_export=test_config.onnx_export)
    yolo_model.num_classes = num_classes
    yolo_model = yolo_model.to(device=test_config.device)
    # Load the pre-trained model weights and fine-tune the model
    if test_config.model_weights_path.endswith(".pth.tar"):
        yolo_model = load_pretrained_torch_state_dict(yolo_model, test_config.model_weights_path)
    elif test_config.model_weights_path.endswith(".weights"):
        load_pretrained_darknet_state_dict(yolo_model, test_config.model_weights_path)
    else:
        yolo_model = load_pretrained_torch_state_dict(yolo_model, test_config.model_weights_path)
    print(f"Loaded `{test_config.model_weights_path}` pretrained model weights successfully.")

    return yolo_model


def test(
        yolo_model: nn.Module,
        test_dataloader: DataLoader,
        names: list,
        iouv: Tensor,
        niou: int,
        config: Any,
        device: torch.device = torch.device("cpu"),
):
    seen = 0

    # Put the model in eval mode
    yolo_model.eval()

    # if test coco91 dataset
    coco91class = coco80_to_coco91_class()

    # Format print information
    s = ("%20s" + "%10s" * 6) % ("Class", "Images", "Targets", "P", "R", "mAP@0.5", "F1")
    p, r, f1, mp, mr, map50, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_index, (images, targets, paths, shapes) in enumerate(tqdm(test_dataloader, desc=s)):
        images = images.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        batch_size, _, height, width = images.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Inference
        with torch.no_grad():
            # Run model
            output, train_out = yolo_model(images, image_augment=config["IMAGE_AUGMENT"])  # inference and training outputs

            # Run NMS
            output = non_max_suppression(output, config["CONF_THRESHOLD"], config["IOU_THRESHOLD"])

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            target_classes = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                  torch.Tensor(),
                                  torch.Tensor(),
                                  target_classes))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if config["SAVE_JSON"]:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split("_")[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(images[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({"image_id": image_id,
                                  "category_id": coco91class[int(p[5])],
                                  "bbox": [round(x, 3) for x in b],
                                  "score": round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
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
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, target_classes)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_classes))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map50, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=yolo_model.num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%10.3g" * 6  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, mf1))

    # Print results per class
    if config["verbose"] and yolo_model.num_classes > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if config["SAVE_JSON"] and map50 and len(jdict):
        print("\nCOCO mAP with pycocotools...")
        imgIds = [int(Path(x).stem.split("_")[-1]) for x in test_dataloader.dataset.image_files]
        with open(config["SAVE_JSON_PATH"], "w") as file:
            json.dump(jdict, file)

        cocoGt = COCO(glob.glob(config["GT_JSON_PATH"])[0])
        cocoDt = cocoGt.loadRes(config["SAVE_JSON_PATH"])  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    # Return results
    maps = np.zeros(yolo_model.num_classes) + map50
    for ap_index, c in enumerate(ap_class):
        maps[c] = ap[ap_index]

    return mp, mr, map50, mf1, maps


if __name__ == "__main__":
    main()
