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
import math
import os
import random
import shutil
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn import functional as F_torch
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from test import test
from yolov3_pytorch.data import LoadImagesAndLabels
from yolov3_pytorch.models import Darknet, compute_loss, load_state_dict, load_resume_state_dict
from yolov3_pytorch.utils import AverageMeter, ProgressMeter, labels_to_class_weights, plot_images
from yolov3_pytorch.utils.common import parse_dataset_config

# Read YAML configuration file
with open("./configs/train/YOLOv3_tiny-VOC.yaml", "r") as f:
    config = yaml.full_load(f)

# Initialize the number of training epochs
start_epoch = 0

# Initialize training to generate network evaluation indicators
best_map50 = 0.0

# Create the folder where the models weights are saved
samples_dir = os.path.join("./samples", config["EXP_NAME"])
results_dir = os.path.join("./results", config["EXP_NAME"])
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# create models training log
writer = SummaryWriter(os.path.join("./samples", "logs", config["EXP_NAME"]))


def main(seed: int):
    # Fixed random number seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Initialize global variables
    global start_epoch, best_map50

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    yolo_model, ema_yolo_model, train_dataloader, test_dataloader, names = build_dataset_and_model(
        config,
        device,
    )
    optimizer = define_optimizer(yolo_model, config)

    # Load the pre-trained models weights and fine-tune the models
    pretrained_model_weights_path = config["TRAIN"]["CHECKPOINT"]["PRETRAINED_MODEL_WEIGHTS_PATH"]
    if pretrained_model_weights_path.endswith(".pth.tar"):
        state_dict = torch.load(pretrained_model_weights_path, map_location=device)["state_dict"]
        yolo_model = load_state_dict(yolo_model, state_dict)
    elif pretrained_model_weights_path.endswith(".weights"):
        yolo_model.load_darknet_weights(pretrained_model_weights_path)
    elif pretrained_model_weights_path != "":
        print("Unsupported pretrained models format. Only support `.pth.tar` and `.weights`.")

    # Load the last training interruption node
    print("Check whether the resume models is restored...")
    resume_model_weights_path = config["TRAIN"]["CHECKPOINT"]["RESUME_MODEL_WEIGHTS_PATH"]
    if resume_model_weights_path.endswith(".pth.tar"):
        yolo_model, ema_yolo_model, start_epoch, best_map50, optimizer = load_resume_state_dict(
            yolo_model,
            resume_model_weights_path,
            ema_yolo_model,
            optimizer,
        )
        print(f"Loaded `{resume_model_weights_path}` resume models weights successfully.")
    else:
        print("Resume training models not found. Start training from scratch.")

    scheduler = define_scheduler(optimizer, start_epoch, config)

    # get the number of training samples
    batches = len(train_dataloader)

    # For test
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()
    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(yolo_model,
              ema_yolo_model,
              train_dataloader,
              optimizer,
              epoch,
              scaler,
              writer,
              batches,
              max(3 * batches, 500),
              device,
              config["TRAIN"]["PRINT_FREQ"])
        p, r, map50, f1 = test(yolo_model,
                               test_dataloader,
                               names,
                               iouv,
                               niou,
                               config,
                               device)
        writer.add_scalar("Test/Precision", p, epoch + 1)
        writer.add_scalar("Test/Recall", r, epoch + 1)
        writer.add_scalar("Test/mAP0.5", map50, epoch + 1)
        writer.add_scalar("Test/F1", f1, epoch + 1)
        print("\n")

        # Update the learning rate after each training epoch
        scheduler.step()

        # Automatically save models weights
        is_best = map50 > best_map50
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_map50 = max(map50, best_map50)
        weights_path = os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar")
        torch.save({"epoch": epoch + 1,
                    "best_map50": best_map50,
                    "state_dict": yolo_model.state_dict(),
                    "ema_state_dict": ema_yolo_model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   weights_path)

        if is_best:
            shutil.copyfile(weights_path, os.path.join(samples_dir, "best.pth.tar"))
        if is_last:
            shutil.copyfile(weights_path, os.path.join(samples_dir, "last.pth.tar"))


def build_dataset_and_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module, DataLoader, DataLoader, list]:
    # Load dataset
    dataset_dict = parse_dataset_config(config["DATASET_CONFIG_PATH"])
    num_classes = 1 if config["SINGLE_CLASSES"] else int(dataset_dict["classes"])
    names = dataset_dict["names"]
    config["TRAIN"]["LOSSES"]["CLS_LOSS"]["WEIGHT"] *= num_classes / 80

    train_datasets = LoadImagesAndLabels(path=dataset_dict["train"],
                                         image_size=config["TRAIN"]["IMG_SIZE_MAX"],
                                         batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                         image_augment=config["TRAIN"]["AUGMENT"],
                                         image_augment_dict=config["AUGMENT_DICT"],
                                         rect_label=config["TRAIN"]["RECT_LABEL"],
                                         cache_images=config["CACHE_IMAGES"],
                                         single_classes=config["SINGLE_CLASSES"],
                                         gray=config["GRAY"])
    test_datasets = LoadImagesAndLabels(path=dataset_dict["test"],
                                        image_size=config["TEST"]["IMG_SIZE"],
                                        batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                        image_augment=config["TEST"]["AUGMENT"],
                                        image_augment_dict=config["AUGMENT_DICT"],
                                        rect_label=config["TEST"]["RECT_LABEL"],
                                        cache_images=config["CACHE_IMAGES"],
                                        single_classes=config["SINGLE_CLASSES"],
                                        pad=0.5,
                                        gray=config["GRAY"])
    # generate dataset iterator
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                  shuffle=not config["TRAIN"]["RECT_LABEL"],
                                  num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                  pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                  drop_last=config["TRAIN"]["HYP"]["DROP_LAST"],
                                  persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"],
                                  collate_fn=train_datasets.collate_fn)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=config["TEST"]["HYP"]["DROP_LAST"],
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"],
                                 collate_fn=test_datasets.collate_fn)

    # Create models
    yolo_model = Darknet(model_config_path=config["MODEL"]["YOLO"]["CONFIG_PATH"],
                         img_size=(416, 416),
                         gray=config["GRAY"],
                         compile_mode=False,
                         onnx_export=config["ONNX_EXPORT"])
    yolo_model = yolo_model.to(device)

    yolo_model.num_classes = num_classes
    yolo_model.image_augment_dict = config["AUGMENT_DICT"]
    yolo_model.gr = 1.0
    yolo_model.class_weights = labels_to_class_weights(train_datasets.labels, 1 if config["SINGLE_CLASSES"] else num_classes)

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average models based on the generator to stabilize models training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_yolo_model = AveragedModel(yolo_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_yolo_model = None

    # 编译模型
    if config["MODEL"]["YOLO"]["COMPILED"]:
        yolo_model = torch.compile(yolo_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_yolo_model is not None:
        ema_yolo_model = torch.compile(ema_yolo_model)

    return yolo_model, ema_yolo_model, train_dataloader, test_dataloader, names


def define_optimizer(yolo_model: nn.Module, config: Any) -> optim.SGD:
    optim_group, weight_decay, biases = [], [], []  # optimizer parameter groups
    for k, v in dict(yolo_model.named_parameters()).items():
        if ".bias" in k:
            biases += [v]  # biases
        elif "Conv2d.weight" in k:
            weight_decay += [v]  # apply weight_decay
        else:
            optim_group += [v]  # all else

    optimizer = optim.SGD(optim_group,
                          lr=config["TRAIN"]["HYP"]["LR"],
                          momentum=config["TRAIN"]["HYP"]["MOMENTUM"],
                          nesterov=config["TRAIN"]["HYP"]["NESTEROV"])
    optimizer.add_param_group({"params": weight_decay, "weight_decay": config["TRAIN"]["HYP"]["WEIGHT_DECAY"]})
    optimizer.add_param_group({"params": biases})
    del optim_group, weight_decay, biases

    return optimizer


def define_scheduler(optimizer: optim.SGD, start_epoch: int, config: Any) -> lr_scheduler.LambdaLR:
    lf = lambda x: (((1 + math.cos(x * math.pi / config["TRAIN"]["HYP"]["EPOCHS"])) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    return scheduler


def train(
        yolo_model: nn.Module,
        ema_yolo_model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        batches: int,
        num_burn: int,
        device: torch.device,
        print_frequency: int = 1,
) -> None:
    """training main function

    Args:
        yolo_model (nn.Module): generator models
        ema_yolo_model (nn.Module): Generator-based exponential mean models
        train_dataloader (DataLoader): training dataset iterator
        optimizer (optim.Adam): optimizer function
        epoch (int): number of training epochs
        scaler (amp.GradScaler): mixed precision function
        writer (SummaryWriter): training log function
        batches (int): number of batches
        num_burn (int): number of burn-in batches
        device (torch.device): PyTorch device
        print_frequency (int, optional): print frequency. Defaults to 1.

    """
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    giou_losses = AverageMeter("GIoULoss", ":6.6f")
    obj_losses = AverageMeter("ObjLoss", ":6.6f")
    cls_losses = AverageMeter("ClsLoss", ":6.6f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, giou_losses, obj_losses, cls_losses, losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    yolo_model.train()

    # Get the initialization training time
    end = time.time()

    # Number of batches to accumulate gradients
    accumulate = max(round(config["TRAIN"]["HYP"]["ACCUMULATE_BATCH_SIZE"] / config["TRAIN"]["HYP"]["IMGS_PER_BATCH"]), 1)

    for batch_index, (images, targets, paths, _) in enumerate(train_dataloader):
        total_batch_index = batch_index + (batches * epoch)
        images = images.to(device).float() / 255.0
        targets = targets.to(device)

        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Training shows
        train_batch_name = config["EXP_NAME"] + ".jpg"
        if total_batch_index < 1:
            if os.path.exists(train_batch_name):
                os.remove(train_batch_name)
            plot_images(images, targets, paths, train_batch_name, max_size=config["TRAIN"]["IMG_SIZE_MAX"])

        # Burn-in
        if total_batch_index <= num_burn:
            xi = [0, num_burn]  # x interp
            yolo_model.gr = np.interp(total_batch_index, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                lr_decay = lambda lr: (((1 + math.cos(lr * math.pi / config["TRAIN"]["HYP"]["EPOCHS"])) / 2) ** 1.0) * 0.95 + 0.05
                x["lr"] = np.interp(total_batch_index, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lr_decay(epoch)])
                x["weight_decay"] = np.interp(total_batch_index,
                                              xi,
                                              [0.0, config["TRAIN"]["HYP"]["WEIGHT_DECAY"] if j == 1 else 0.0])
                if "momentum" in x:
                    x["momentum"] = np.interp(total_batch_index, xi, [0.9, config["TRAIN"]["HYP"]["MOMENTUM"]])

        # Multi-Scale
        image_size = random.randrange(config["TRAIN"]["IMG_SIZE_MIN"] // config["TRAIN"]["GRID_SIZE"],
                                      config["TRAIN"]["IMG_SIZE_MAX"] // config["TRAIN"]["GRID_SIZE"] + 1) * config["TRAIN"]["GRID_SIZE"]
        scale_factor = image_size / max(images.shape[2:])  # scale factor
        if scale_factor != 1:
            # new shape (stretched to 32-multiple)
            new_image_size = [math.ceil(x * scale_factor / config["TRAIN"]["GRID_SIZE"]) * config["TRAIN"]["GRID_SIZE"] for x in
                              images.shape[2:]]
            images = F_torch.interpolate(images, size=new_image_size, mode="bilinear", align_corners=False)

        # Initialize the generator gradient
        yolo_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = yolo_model(images)
            loss, loss_item = compute_loss(output,
                                           targets,
                                           yolo_model,
                                           config["TRAIN"]["HYP"]["IOU_THRESHOLD"],
                                           config["TRAIN"]["LOSSES"])
            loss *= config["TRAIN"]["HYP"]["IMGS_PER_BATCH"] / config["TRAIN"]["HYP"]["ACCUMULATE_BATCH_SIZE"]

        # Backpropagation
        scaler.scale(loss).backward()

        # update generator weights
        if total_batch_index % accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

        # update exponential average models weights
        ema_yolo_model.update_parameters(yolo_model)

        # Statistical loss value for terminal data output
        giou_losses.update(loss_item[0], images.size(0))
        obj_losses.update(loss_item[1], images.size(0))
        cls_losses.update(loss_item[2], images.size(0))
        losses.update(loss_item[3], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record training log information
        if batch_index % print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar("Train/GIoULoss", loss_item[0], total_batch_index)
            writer.add_scalar("Train/ObjLoss", loss_item[1], total_batch_index)
            writer.add_scalar("Train/ClsLoss", loss_item[2], total_batch_index)
            writer.add_scalar("Train/Loss", loss_item[3], total_batch_index)

            progress.display(batch_index)


if __name__ == "__main__":
    main(config["SEED"])
