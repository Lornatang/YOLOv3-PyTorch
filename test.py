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
import argparse
import glob
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Darknet
from models import load_darknet_weights
from utils import LoadImagesAndLabels
from utils import ap_per_class
from utils import box_iou
from utils import clip_coords
from utils import coco80_to_coco91_class
from utils import compute_loss
from utils import load_classes
from utils import non_max_suppression
from utils import parse_data_config
from utils import scale_coords
from utils import scale_image
from utils import select_device
from utils import time_synchronized
from utils import xywh2xyxy
from utils import xyxy2xywh


def evaluate(cfg,
             data,
             weights=None,
             batch_size=16,
             workers=4,
             image_size=416,
             confidence_threshold=0.001,
             iou_threshold=0.6,  # for nms
             save_json=True,
             single_cls=False,
             model=None,
             dataloader=None):
    # Initialize/load model and set device
    if model is None:
        device = select_device(args.device, batch_size=batch_size)
        verbose = args.task == "eval"

        # Initialize model
        model = Darknet(cfg, image_size).to(device)

        # Load weights
        if weights.endswith(".pth"):
            model.load_state_dict(torch.load(weights, map_location=device)["model"])
        else:
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != "cpu" and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_config(data)
    classes_num = 1 if single_cls else int(data["classes"])
    path = data["valid"]  # path to valid images
    names = load_classes(data["names"])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, image_size, batch_size, rect=True)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=workers,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    json_dict, stats, ap, ap_class = [], [], [], []
    for batch_i, (images, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        images = images.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        batch_size, _, height, width = images.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            augment = False  # augment https://github.com/ultralytics/yolov3/issues/931
            if augment:
                images = torch.cat((images, images.flip(3), scale_image(images, 0.7),), 0)

            # Run model
            t = time_synchronized()
            inference_outputs, training_outputs = model(images)
            t0 += time_synchronized() - t

            if augment:
                x = torch.split(inference_outputs, batch_size, dim=0)
                x[1][..., 0] = width - x[1][..., 0]  # flip lr
                x[2][..., :4] /= 0.7  # scale
                inference_outputs = torch.cat(x, 1)

            # Compute loss
            if hasattr(model, "hyp"):  # if model has loss hyperparameters
                loss += compute_loss(training_outputs, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inference_outputs, conf_thres=confidence_threshold, iou_thres=iou_threshold)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            label_num = len(labels)
            target_class = labels[:, 0].tolist() if label_num else []
            seen += 1

            if pred is None:
                if label_num:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_class))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(images[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    json_dict.append({'image_id': image_id,
                                      'category_id': coco91class[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(len(pred), niou, dtype=torch.bool, device=device)
            if label_num:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == label_num:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=classes_num)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    context = "%20s" + "%10.3g" * 6  # print format
    print(context % ("all", seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and classes_num > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(context % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (image_size, image_size, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(json_dict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.image_files]
        with open('results.json', 'w') as file:
            json.dump(json_dict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO(glob.glob('data/coco2014/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    maps = np.zeros(classes_num) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="cfgs/yolov3.cfg",
                        help="Neural network profile path. (default=cfgs/yolov3.cfg)")
    parser.add_argument("--data", type=str, default="cfgs/coco2014.data",
                        help="Dataload load path. (default=data/coco2014.data)")
    parser.add_argument("--weight", type=str, default="weights/yolov3.pth",
                        help="Model file weight path. (default=weights/yolov3.pth")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Size of each image batch. (default=32)")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument("--image-size", type=int, default=416,
                        help="Size of processing picture. (default=416)")
    parser.add_argument("--confidence-threshold", type=float, default=0.001,
                        help="Object confidence threshold. (default=0.001)")
    parser.add_argument("--iou-threshold", type=float, default=0.6,
                        help="IOU threshold for NMS. (default=0.6)")
    parser.add_argument("--task", default="eval", help="`eval`, `study`, `benchmark`")
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1) or cpu")
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument("--single-cls", action="store_true", help="train as single-class dataset")
    args = parser.parse_args()

    print(args)

    # task = "eval", "study", "benchmark"
    if args.task == "eval":  # (default) eval normally
        evaluate(args.cfg,
                 args.data,
                 args.weight,
                 args.batch_size,
                 args.workers,
                 args.image_size,
                 args.confidence_threshold,
                 args.iou_threshold,
                 args.save_json,
                 args.single_cls)

    elif args.task == "benchmark":  # mAPs at 320-608 at conf 0.5 and 0.7
        y = []
        for i in [320, 416, 512, 608]:  # img-size
            for j in [0.5, 0.7]:  # iou-thres
                t = time.time()
                r = evaluate(args.cfg, args.data, args.weights, args.batch_size, i, args.confidence_threshold, j)[0]
                y.append(r + (time.time() - t,))
        np.savetxt("benchmark.txt", y, fmt="%10.4g")  # y = np.loadtxt("study.txt")

    elif args.task == "study":  # Parameter study
        y = []
        x = np.arange(0.4, 0.9, 0.05)  # iou-thres
        for i in x:
            t = time.time()
            r = evaluate(args.cfg, args.data, args.weights, args.batch_size, args.image_size, args.confidence_threshold,
                         i)[0]
            y.append(r + (time.time() - t,))
        np.savetxt("study.txt", y, fmt="%10.4g")  # y = np.loadtxt("study.txt")

        # Plot
        fig, ax = plt.subplots(3, 1, figsize=(6, 6))
        y = np.stack(y, 0)
        ax[0].plot(x, y[:, 2], marker=".", label="mAP@0.5")
        ax[0].set_ylabel("mAP")
        ax[1].plot(x, y[:, 3], marker=".", label="mAP@0.5:0.95")
        ax[1].set_ylabel("mAP")
        ax[2].plot(x, y[:, -1], marker=".", label="time")
        ax[2].set_ylabel("time (s)")
        for i in range(3):
            ax[i].legend()
            ax[i].set_xlabel("iou_threshold")
        fig.tight_layout()
        plt.savefig("study.jpg", dpi=200)
