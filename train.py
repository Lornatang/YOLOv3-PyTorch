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
import time

import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn import functional as F_torch
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
import train_config
from dataset import parse_dataset_config, labels_to_class_weights, LoadImagesAndLabels
from test import test
from utils import load_pretrained_torch_state_dict, load_pretrained_darknet_state_dict, load_resume_torch_state_dict, \
    save_torch_state_dict, make_directory, AverageMeter, ProgressMeter, plot_images


def main():
    device = torch.device(train_config.device)
    # Fixed random number seed
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_map50 = 0.0

    yolo_model, ema_yolo_model, train_dataloader, test_dataloader, names = build_dataset_and_model(
        train_config.model_ema_decay,
        device,
    )
    optimizer = define_optimizer(yolo_model)

    # Load the pre-trained model weights and fine-tune the model
    print("Check whether to load pretrained model weights...")
    if train_config.pretrained_model_weights_path.endswith(".pth.tar"):
        yolo_model = load_pretrained_torch_state_dict(yolo_model, train_config.pretrained_model_weights_path)
        print(f"Loaded `{train_config.pretrained_model_weights_path}` pretrained model weights successfully.")
    elif train_config.pretrained_model_weights_path.endswith(".weights"):
        load_pretrained_darknet_state_dict(yolo_model, train_config.pretrained_model_weights_path)
        print(f"Loaded `{train_config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption node
    print("Check whether the resume model is restored...")
    if train_config.resume_model_weights_path.endswith(".pth.tar"):
        yolo_model, ema_yolo_model, start_epoch, best_map50, optimizer = load_resume_torch_state_dict(
            yolo_model,
            train_config.resume_model_weights_path,
            ema_yolo_model,
            optimizer,
        )
        print(f"Loaded `{train_config.resume_model_weights_path}` resume model weights successfully.")
    else:
        print("Resume training model not found. Start training from scratch.")

    scheduler = define_scheduler(optimizer, start_epoch, train_config.epochs)

    # Model weight save address
    samples_dir = os.path.join("samples", train_config.exp_name)
    results_dir = os.path.join("results", train_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", train_config.exp_name))

    # get the number of training samples
    batches = len(train_dataloader)

    # For test
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()
    for epoch in range(start_epoch, train_config.epochs):
        train(yolo_model,
              ema_yolo_model,
              train_dataloader,
              optimizer,
              epoch,
              scaler,
              writer,
              batches,
              max(3 * batches, 500),
              train_config.train_print_frequency)
        is_last = (epoch + 1) == train_config.epochs
        p, r, map50, f1, maps = test(yolo_model,
                                     test_dataloader,
                                     names,
                                     train_config.conf_threshold,
                                     train_config.iou_threshold,
                                     is_last and train_config.save_json,
                                     train_config.test_augment,
                                     iouv,
                                     niou,
                                     train_config.verbose,
                                     device)
        writer.add_scalar("Test/Precision", p, epoch + 1)
        writer.add_scalar("Test/Recall", r, epoch + 1)
        writer.add_scalar("Test/mAP0.5", map50, epoch + 1)
        writer.add_scalar("Test/F1", f1, epoch + 1)
        print("\n")

        # Update the learning rate after each training epoch
        scheduler.step()

        # Automatically save model weights
        is_best = map50 > best_map50
        is_last = (epoch + 1) == train_config.epochs
        best_map50 = max(map50, best_map50)
        save_torch_state_dict({"epoch": epoch + 1,
                               "best_map50": best_map50,
                               "state_dict": yolo_model.state_dict(),
                               "ema_state_dict": ema_yolo_model.state_dict(),
                               "optimizer": optimizer.state_dict()},
                              f"epoch_{epoch + 1}.pth.tar",
                              samples_dir,
                              results_dir,
                              "best.pth.tar",
                              "last.pth.tar",
                              is_best,
                              is_last)


def build_dataset_and_model(
        model_ema_decay: float,
        device: torch.device,
) -> [nn.Module, nn.Module, DataLoader, DataLoader, list]:
    # Load dataset
    dataset_dict = parse_dataset_config(train_config.dataset_config_path)
    num_classes = 1 if train_config.single_classes else int(dataset_dict["classes"])
    names = dataset_dict["names"]
    train_config.hyper_parameters_dict["cls"] *= num_classes / 80

    train_datasets = LoadImagesAndLabels(path=dataset_dict["train"],
                                         image_size=train_config.train_image_size_max,
                                         batch_size=train_config.batch_size,
                                         augment=train_config.train_augment,
                                         hyper_parameters_dict=train_config.hyper_parameters_dict,
                                         rect_label=train_config.train_rect_label,
                                         cache_images=train_config.cache_images,
                                         single_classes=train_config.single_classes,
                                         gray=train_config.gray)
    test_datasets = LoadImagesAndLabels(path=dataset_dict["test"],
                                        image_size=train_config.test_image_size,
                                        batch_size=train_config.batch_size,
                                        augment=train_config.test_augment,
                                        hyper_parameters_dict=train_config.hyper_parameters_dict,
                                        rect_label=train_config.test_rect_label,
                                        cache_images=train_config.cache_images,
                                        single_classes=train_config.single_classes,
                                        gray=train_config.gray)
    # generate dataset iterator
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=train_config.batch_size,
                                  shuffle=not train_config.train_rect_label,
                                  num_workers=train_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True,
                                  collate_fn=train_datasets.collate_fn)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=train_config.batch_size,
                                 shuffle=False,
                                 num_workers=train_config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True,
                                 collate_fn=train_datasets.collate_fn)

    # Create model
    yolo_model = model.__dict__[train_config.model_arch_name](image_size=(416, 416),
                                                              gray=train_config.gray,
                                                              onnx_export=train_config.onnx_export)
    yolo_model = yolo_model.to(device)

    yolo_model.num_classes = num_classes
    yolo_model.hyper_parameters_dict = train_config.hyper_parameters_dict
    yolo_model.gr = 1.0
    yolo_model.class_weights = labels_to_class_weights(train_datasets.labels,
                                                       1 if train_config.single_classes else num_classes)

    # Generate an exponential average model based on the generator to stabilize model training
    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
        (1 - model_ema_decay) * averaged_model_parameter + model_ema_decay * model_parameter
    ema_yolo_model = AveragedModel(yolo_model, device=device, avg_fn=ema_avg_fn)

    ema_yolo_model = ema_yolo_model.to(device)

    return yolo_model, ema_yolo_model, train_dataloader, test_dataloader, names


def define_optimizer(yolo_model: nn.Module) -> optim.SGD:
    optim_group, weight_decay, biases = [], [], []  # optimizer parameter groups
    for k, v in dict(yolo_model.named_parameters()).items():
        if ".bias" in k:
            biases += [v]  # biases
        elif "Conv2d.weight" in k:
            weight_decay += [v]  # apply weight_decay
        else:
            optim_group += [v]  # all else

    optimizer = optim.SGD(optim_group,
                          lr=train_config.optim_lr,
                          momentum=train_config.optim_momentum,
                          nesterov=True)
    optimizer.add_param_group({"params": weight_decay, "weight_decay": train_config.optim_weight_decay})
    optimizer.add_param_group({"params": biases})
    del optim_group, weight_decay, biases

    return optimizer


def define_scheduler(optimizer: optim.SGD, start_epoch: int, epochs: int) -> lr_scheduler.LambdaLR:
    """
    Define the learning rate scheduler

    Paper:
        https://arxiv.org/pdf/1812.01187.pdf

    Args:
        optimizer (optim.SGD): The optimizer to be used for training
        start_epoch (int): The epoch to start training from
        epochs (int): The total number of epochs to train for

    Returns:
        lr_scheduler.LambdaLR: The learning rate scheduler
        
    """
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
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
        print_frequency: int = 1,
) -> None:
    """training main function

    Args:
        yolo_model (nn.Module): generator model
        ema_yolo_model (nn.Module): Generator-based exponential mean model
        train_dataloader (DataLoader): training dataset iterator
        optimizer (optim.Adam): optimizer function
        epoch (int): number of training epochs
        scaler (amp.GradScaler): mixed precision function
        writer (SummaryWriter): training log function
        batches (int): number of batches
        num_burn (int): number of burn-in batches
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
    accumulate = max(round(train_config.accumulate_batch_size / train_config.batch_size), 1)

    for batch_index, (images, targets, paths, _) in enumerate(train_dataloader):
        total_batch_index = batch_index + (batches * epoch) + 1
        images = images.to(train_config.device).float() / 255.0
        targets = targets.to(train_config.device)

        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        if total_batch_index < 2:
            plot_images(images=images, targets=targets, paths=paths, file_name=f"train_batch_{total_batch_index}.jpg")

        # Burn-in
        if total_batch_index <= num_burn:
            xi = [0, num_burn]  # x interp
            model.gr = np.interp(total_batch_index, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                lr_decay = lambda lr: (((1 + math.cos(lr * math.pi / train_config.epochs)) / 2) ** 1.0) * 0.95 + 0.05
                x["lr"] = np.interp(total_batch_index, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lr_decay(epoch)])
                x["weight_decay"] = np.interp(total_batch_index,
                                              xi,
                                              [0.0, train_config.optim_weight_decay if j == 1 else 0.0])
                if "momentum" in x:
                    x["momentum"] = np.interp(total_batch_index, xi, [0.9, train_config.optim_momentum])

        # Multi-Scale
        image_size = random.randrange(train_config.train_image_size_min // train_config.grid_size,
                                      train_config.train_image_size_max // train_config.grid_size + 1) * train_config.grid_size
        scale_factor = image_size / max(images.shape[2:])  # scale factor
        if scale_factor != 1:
            # new shape (stretched to 32-multiple)
            new_image_size = [math.ceil(x * scale_factor / train_config.grid_size) * train_config.grid_size for x in
                              images.shape[2:]]
            images = F_torch.interpolate(images, size=new_image_size, mode="bilinear", align_corners=False)

        # Initialize the generator gradient
        yolo_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = yolo_model(images)
            loss, loss_item = model.compute_loss(output, targets, yolo_model)
            loss *= train_config.batch_size / train_config.accumulate_batch_size

        # Backpropagation
        scaler.scale(loss).backward()

        # update generator weights
        if total_batch_index % accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

        # update exponential average model weights
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

            progress.display(batch_index + 1)


if __name__ == "__main__":
    main()
