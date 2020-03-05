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
import os
import random
import time

import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import warnings
from tqdm import tqdm

from models import Darknet
from models import load_darknet_weights
from test import evaluate
from utils import LoadImagesAndLabels
from utils import compute_loss
from utils import fitness
from utils import init_seeds
from utils import labels_to_class_weights
from utils import labels_to_image_weights
from utils import model_info
from utils import parse_data_cfg
from utils import plot_results
from utils import print_model_biases
from utils import select_device

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except ImportWarning:
    mixed_precision = False  # not installed

parameters = {"giou": 3.54,  # giou loss gain
              "cls": 37.4,  # cls loss gain
              "cls_pw": 1.0,  # cls BCELoss positive_weight
              "obj": 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
              "obj_pw": 1.0,  # obj BCELoss positive_weight
              "iou_t": 0.225,  # iou training threshold
              'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
              "lrf": -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
              "momentum": 0.937,  # SGD momentum
              "weight_decay": 0.000484,  # optimizer weight decay
              "fl_gamma": 0.5,  # focal loss gamma
              "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
              "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
              "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
              "degrees": 1.98,  # image rotation (+/- deg)
              "translate": 0.05,  # image translation (+/- fraction)
              "scale": 0.05,  # image scale (+/- gain)
              "shear": 0.641}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt
parameter_file = glob.glob("hyp*.txt")
if parameter_file:
    print(f"Using {parameter_file[0]}")
    for keys, value in zip(parameters.keys(), np.loadtxt(parameter_file[0])):
        parameters[keys] = value


def train():
    cfg = args.cfg
    data = args.data
    image_size, img_size_val = args.image_size if len(
        args.image_size) == 2 else args.image_size * 2  # train, test sizes
    epochs = args.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = args.batch_size
    accumulate = args.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = args.weights  # initial training weights

    # Initialize
    init_seeds()
    if args.multi_scale:
        img_sz_min = round(image_size / 32 / 1.5)
        img_sz_max = round(image_size / 32 * 1.5)
        image_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print(f"Using multi-scale {img_sz_min * 32} - {image_size}")

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    valid_path = data_dict["valid"]
    nc = 1 if args.single_cls else int(data_dict["classes"])  # number of classes

    # Remove previous results
    for files in glob.glob("results.txt"):
        os.remove(files)

    # Initialize model
    model = Darknet(cfg, arch=args.arch).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if ".bias" in k:
            pg2 += [v]  # biases
        elif "Conv2d.weight" in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    optimizer = torch.optim.SGD(pg0, lr=parameters["lr0"], momentum=parameters["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": parameters["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    optimizer.param_groups[2]["lr"] *= 2.0  # bias lr
    del pg0, pg1, pg2

    epoch = 0
    start_epoch = 0
    best_fitness = 0.0
    if weights.endswith(".pth"):
        # possible weights are "*.pth", "yolov3-spp.pth", "yolov3-tiny.pth" etc.
        state = torch.load(weights, map_location=device)

        # load model
        try:
            state["model"] = {k: v for k, v in state["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(state["model"], strict=False)
        except KeyError as e:
            s = f"{args.weights} is not compatible with {args.cfg}. Specify --weights "" or specify a --cfg compatible with {args.weights}. "
            raise KeyError(s) from e

        # load optimizer
        if state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
            best_fitness = state["best_fitness"]

        # load results
        if state.get("training_results") is not None:
            with open("results.txt", "w") as file:
                file.write(state["training_results"])  # write results.txt

        start_epoch = state["epoch"] + 1
        del state

    elif len(weights) > 0:
        # possible weights are "*.weights", "yolov3-tiny.conv.15",  "darknet53.conv.74" etc.
        load_darknet_weights(model, weights)
    else:
        print("Pre training model weight not loaded")

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    lf = lambda x: (1 + math.cos(x * math.pi / epochs)) / 2 * 0.99 + 0.01  # cosine https://arxiv.org/pdf/1812.01187.pdf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch

    # Initialize distributed training
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend="nccl",  # "distributed backend"
                                init_method="tcp://127.0.0.1:9999",  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    train_dataset = LoadImagesAndLabels(train_path, image_size, batch_size,
                                        augment=True,
                                        hyp=parameters,  # augmentation hyperparameters
                                        rect=args.rect,  # rectangular training
                                        cache_images=args.cache_images,
                                        single_cls=args.single_cls)
    valid_dataset = LoadImagesAndLabels(valid_path, img_size_val, batch_size * 2,
                                        augment=False,
                                        hyp=parameters,  # no apply augmentation hyperparameters
                                        rect=True,  # rectangular inference
                                        cache_images=args.cache_images,
                                        single_cls=args.single_cls)

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=not args.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size * 2,
                                                   num_workers=args.workers,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    # Start training
    nb = len(train_dataloader)
    prebias = start_epoch == 0
    model.nc = nc  # attach number of classes to model
    model.arch = args.arch  # attach yolo architecture
    model.hyp = parameters  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # "P", "R", "mAP", "F1", "val GIoU", "val Objectness", "val Classification"
    model_info(model, report="summary")  # "full" or "summary"
    print(f"Using {args.workers} dataloader workers.")
    print(f"Starting training for {args.epochs} epochs...")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.hyp['gr'] = 1 - (1 + math.cos(min(epoch * 2, epochs) * math.pi / epochs)) / 2  # GIoU <-> 1.0 loss ratio

        # Prebias
        if prebias:
            ne = max(round(30 / nb), 3)  # number of prebias epochs
            # prebias settings (lr=0.1, momentum=0.9)
            ps = np.interp(epoch, [0, ne], [0.1, parameters["lr0"] * 2]), np.interp(epoch, [0, ne],
                                                                                    [0.9, parameters["momentum"]])
            if epoch == ne:
                print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]["lr"] = ps[0]
            if optimizer.param_groups[2].get("momentum") is not None:  # for SGD but not Adam
                optimizer.param_groups[2]["momentum"] = ps[1]

        # Update image weights (optional)
        if train_dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=w)
            # rand weighted idx
            train_dataset.indices = random.choices(range(train_dataset.image_files_num), weights=image_weights,
                                                   k=train_dataset.image_files_num)

        mean_losses = torch.zeros(4).to(device)
        print(("\n" + "%10s" * 8) % ("Epoch", "memory", "GIoU", "obj", "cls", "total", "targets", "image_size"))
        progress_bar = tqdm(enumerate(train_dataloader), total=nb)
        for i, (images, targets, paths, _) in progress_bar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            images = images.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Multi-Scale training
            if args.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    image_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = image_size / max(images.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in
                          images.shape[2:]]  # new shape (stretched to 32-multiple)
                    images = F.interpolate(images, size=ns, mode="bilinear", align_corners=False)

            # Run model
            output = model(images)

            # Compute loss
            loss, loss_items = compute_loss(output, targets, model)
            if not torch.isfinite(loss):
                warnings.warn(f"WARNING: Non-finite loss, ending training {loss_items}")
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mean_losses = (mean_losses * i + loss_items) / (i + 1)  # update mean losses
            memory = f"{torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0:.2f}G"
            s = ("%10s" * 2 + "%10.3g" * 6) % (
                "%g/%g" % (epoch, args.epochs - 1), memory, *mean_losses, len(targets), image_size)
            progress_bar.set_description(s)

        # Process epoch results
        results, maps = evaluate(cfg,
                                 data,
                                 batch_size=batch_size * 2,
                                 image_size=img_size_val,
                                 model=model,
                                 confidence_threshold=0.001,
                                 iou_threshold=0.6,
                                 single_cls=args.single_cls,
                                 dataloader=valid_dataloader)

        # Update scheduler
        scheduler.step(epoch)

        # Write epoch results
        with open("results.txt", "a") as f:
            f.write(s + "%10.3g" * 7 % results + "\n")  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Write Tensorboard results
        if tb_writer:
            x = list(mean_losses) + list(results)
            titles = ["GIoU", "Objectness", "Classification", "Train loss",
                      "Precision", "Recall", "mAP", "F1", "val GIoU", "val Objectness", "val Classification"]
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fitness_i = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fitness_i > best_fitness:
            best_fitness = fitness_i

        # Save training results
        with open("results.txt", "r") as f:
            # Create checkpoint
            state = {"epoch": epoch,
                     "best_fitness": best_fitness,
                     "training_results": f.read(),
                     "model": model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel
                     else model.state_dict(),
                     "optimizer": optimizer.state_dict()}

        # Save last checkpoint
        torch.save(state, "weights/checkpoint.pth")

        # Save best checkpoint
        if best_fitness == fitness_i:
            torch.save(state, "weights/model_best.pth")

        # Delete checkpoint
        del state

    plot_results()
    print(f"{epoch - start_epoch} epochs completed in {(time.time() - start_time) / 3600:.3f} hours.\n")
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=273,
                        help="Note: 500200 batches at bs 16, 117263 COCO images = 273 epochs. (default=273)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Note: Effective bs = batch_size * accumulate = 16 * 4 = 64. (default=16)")
    parser.add_argument("--accumulate", type=int, default=4,
                        help="Batches to accumulate before optimizing. (default=4)")
    parser.add_argument("--cfg", type=str, default="cfg/yolov3.cfg",
                        help="Neural network profile path. (default=`cfg/yolov3.cfg`)")
    parser.add_argument("--data", type=str, default="data/coco2014.data",
                        help="Dataload load path. (default=`cfg/coco2014.data`)")
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument("--multi-scale", action="store_true", help="adjust (67% - 150%) img_size every 10 batches")
    parser.add_argument("--image-size", nargs='+', type=int, default=[608],
                        help="Size of processing picture. (default=[608])")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", action="store_true", help="resume training from last.pth")
    parser.add_argument("--cache-images", action="store_true", help="cache images for faster training")
    parser.add_argument("--weights", type=str, default="",
                        help="Model file weight path. (default=``)")
    parser.add_argument("--arch", type=str, default="default",
                        help="Yolo architecture. (default=`default`)")
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1 or cpu)")
    parser.add_argument("--single-cls", action="store_true", help="train as single-class dataset")
    args = parser.parse_args()
    print(args)

    if args.resume:
        args.weights = args.resume

    device = select_device(args.device, apex=mixed_precision, batch_size=args.batch_size)
    if device.type == "cpu":
        mixed_precision = False

    try:
        os.makedirs("weights")
    except OSError:
        pass

    try:
        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
    except ImportWarning:
        pass

    tb_writer = SummaryWriter()

    train()
