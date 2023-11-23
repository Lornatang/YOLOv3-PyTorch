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
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import optim
from torch.cuda import amp
from torch.nn import functional as F_torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from yolov3_pytorch.data.base import BaseDatasets
from yolov3_pytorch.engine.evaler import Evaler
from yolov3_pytorch.models.darknet import Darknet
from yolov3_pytorch.models.losses import compute_loss
from yolov3_pytorch.models.utils import load_state_dict, load_resume_state_dict
from yolov3_pytorch.utils.common import labels_to_class_weights
from yolov3_pytorch.utils.loggers import AverageMeter, ProgressMeter
from yolov3_pytorch.utils.plots import plot_images

__all__ = [
    "Trainer",
]


class Trainer:
    def __init__(
            self,
            config: Dict,
            scaler: amp.GradScaler,
            device: torch.device,
            save_weights_dir: str | Path,
            tblogger: SummaryWriter,
    ) -> None:
        self.config = config
        self.scaler = scaler
        self.device = device
        self.save_weights_dir = save_weights_dir
        self.tblogger = tblogger

        self.start_epoch = 0
        self.best_mean_ap = 0.0
        self.evaler = Evaler(self.config, self.device)

        self.train_datasets, self.test_datasets, self.train_dataloader, self.test_dataloader = self.load_datasets()
        self.train_batches, self.test_batches = len(self.train_dataloader), len(self.test_dataloader)
        self.model, self.ema_model = self.build_model()
        self.optimizer = self.define_optimizer()
        self.scheduler = self.define_scheduler()

    def load_datasets(self) -> tuple:
        r"""Load training and test datasets from a configuration file, such as yaml

        Returns:
            tuple: train_dataloader, test_dataloader

        """
        if self.config["DATASET"]["SINGLE_CLASSES"] == 1:
            self.config["MODEL"]["NUM_CLASSES"] = 1
        self.config["TRAIN"]["LOSSES"]["CLS_LOSS"]["WEIGHT"] *= self.config["MODEL"]["NUM_CLASSES"] / 80

        train_datasets = BaseDatasets(self.config["DATASET"]["TRAIN_PATH"],
                                      self.config["TRAIN"]["IMG_SIZE"],
                                      self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                      self.config["AUGMENT"]["ENABLE"],
                                      self.config["AUGMENT"]["HYP"],
                                      self.config["DATASET"]["RECT_LABEL"],
                                      self.config["DATASET"]["CACHE_IMAGES"],
                                      self.config["DATASET"]["SINGLE_CLASSES"],
                                      pad=0.0,
                                      gray=self.config["MODEL"]["GRAY"])
        test_datasets = BaseDatasets(self.config["DATASET"]["TEST_PATH"],
                                     self.config["TEST"]["IMG_SIZE"],
                                     self.config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                     False,
                                     self.config["AUGMENT"]["HYP"],
                                     self.config["DATASET"]["RECT_LABEL"],
                                     self.config["DATASET"]["CACHE_IMAGES"],
                                     self.config["DATASET"]["SINGLE_CLASSES"],
                                     pad=0.5,
                                     gray=self.config["MODEL"]["GRAY"])
        train_dataloader = DataLoader(train_datasets,
                                      batch_size=self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                      shuffle=True,
                                      num_workers=4,
                                      pin_memory=True,
                                      drop_last=True,
                                      persistent_workers=True,
                                      collate_fn=train_datasets.collate_fn)
        test_dataloader = DataLoader(test_datasets,
                                     batch_size=self.config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     drop_last=False,
                                     persistent_workers=True,
                                     collate_fn=test_datasets.collate_fn)

        return train_datasets, test_datasets, train_dataloader, test_dataloader

    def build_model(self) -> tuple:
        # Create model
        model = Darknet(self.config["MODEL"]["CONFIG_PATH"],
                        self.config["TRAIN"]["IMG_SIZE"],
                        self.config["MODEL"]["GRAY"],
                        self.config["MODEL"]["COMPILED"],
                        False)
        model = model.to(self.device)

        model.num_classes = self.config["MODEL"]["NUM_CLASSES"]
        model.class_weights = labels_to_class_weights(self.train_datasets.labels, model.num_classes)

        # Generate an exponential average models based on the generator to stabilize models training
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.001 * averaged_model_parameter + 0.999 * model_parameter
        ema_model = AveragedModel(model, self.device, ema_avg_fn)

        # Compile model
        if self.config["MODEL"]["COMPILED"]:
            model = torch.compile(model)
            ema_model = torch.compile(ema_model)

        return model, ema_model

    def define_optimizer(self) -> optim:
        optim_group, weight_decay, biases = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if ".bias" in k:
                biases += [v]  # biases
            elif "Conv2d.weight" in k:
                weight_decay += [v]  # apply weight_decay
            else:
                optim_group += [v]  # all else

        if self.config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
            optimizer = optim.Adam(optim_group,
                                   self.config["TRAIN"]["OPTIM"]["LR"],
                                   (self.config["TRAIN"]["OPTIM"]["BETA1"], self.config["TRAIN"]["OPTIM"]["BETA2"]),
                                   weight_decay=self.config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
        elif self.config["TRAIN"]["OPTIM"]["NAME"] == "SGD":
            optimizer = optim.SGD(optim_group,
                                  self.config["TRAIN"]["OPTIM"]["LR"],
                                  self.config["TRAIN"]["OPTIM"]["MOMENTUM"],
                                  weight_decay=self.config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
            optimizer.add_param_group({"params": weight_decay, "weight_decay": self.config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"]})
            optimizer.add_param_group({"params": biases})
        else:
            raise NotImplementedError("Only Support ['SGD', 'Adam'] optimizer")
        del optim_group, weight_decay, biases

        return optimizer

    def define_scheduler(self) -> optim.lr_scheduler:
        r"""Define a learning rate scheduler

        Returns:
            torch.optim.lr_scheduler: learning rate scheduler
        """

        # Only use SGD optimizer
        if self.config["TRAIN"]["OPTIM"]["LR_SCHEDULER"]["NAME"] == "LambdaLR" and self.config["TRAIN"]["OPTIM"]["NAME"] == "SGD":
            lf = lambda x: (((1 + math.cos(x * math.pi / self.config["TRAIN"]["HYP"]["EPOCHS"])) / 2) ** 1.0) * 0.95 + 0.05  # cosine
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
            scheduler.last_epoch = self.start_epoch - 1
        else:
            print("No learning rate scheduler")
            scheduler = None

        return scheduler

    def load_checkpoint(self) -> None:
        # Load weights
        pretrained_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["PRETRAINED_MODEL_WEIGHTS_PATH"]
        resume_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["RESUME_MODEL_WEIGHTS_PATH"]

        if pretrained_model_weights_path != "" and os.path.exists(pretrained_model_weights_path):
            print(f"Load pretrained model weights from '{pretrained_model_weights_path}'")
            state_dict = torch.load(pretrained_model_weights_path, map_location=self.device)["state_dict"]
            self.model = load_state_dict(self.model, state_dict)
        elif resume_model_weights_path != "" and os.path.exists(resume_model_weights_path):
            print(f"Load resume model weights from '{resume_model_weights_path}'")
            self.start_epoch, self.best_mean_ap, self.model, self.ema_model, self.optimizer = load_resume_state_dict(self.model,
                                                                                                                     self.ema_model,
                                                                                                                     resume_model_weights_path)
        else:
            print("No pretrained or resume model weights. Train from scratch")

    def save_checkpoint(self, epoch: int, mean_ap: float) -> None:
        # Automatically save models weights
        is_best = mean_ap > self.best_mean_ap
        is_last = (epoch + 1) == self.config["TRAIN"]["HYP"]["EPOCHS"]
        self.best_mean_ap = max(mean_ap, self.best_mean_ap)
        weights_path = os.path.join(self.save_weights_dir, f"epoch_{epoch}.pth.tar")
        torch.save({"epoch": epoch + 1,
                    "best_mean_ap": self.best_mean_ap,
                    "state_dict": self.model.state_dict(),
                    "ema_state_dict": self.ema_model.state_dict(),
                    "optimizer": self.optimizer},
                   weights_path)
        if is_best:
            best_weights_path = os.path.join(self.save_weights_dir, "best.pth.tar")
            shutil.copyfile(weights_path, best_weights_path)
        if is_last:
            last_weights_path = os.path.join(self.save_weights_dir, "last.pth.tar")
            shutil.copyfile(weights_path, last_weights_path)

    def train_on_epoch(self, epoch):
        # The information printed by the progress bar
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        giou_losses = AverageMeter("GIoULoss", ":6.6f")
        obj_losses = AverageMeter("ObjLoss", ":6.6f")
        cls_losses = AverageMeter("ClsLoss", ":6.6f")
        losses = AverageMeter("Loss", ":6.6f")
        progress = ProgressMeter(self.train_batches,
                                 [batch_time, data_time, giou_losses, obj_losses, cls_losses, losses],
                                 prefix=f"Epoch: [{epoch + 1}]")

        # Put the generator in training mode
        self.model.train()

        # Get the initialization training time
        end = time.time()

        # Number of batches to accumulate gradients
        accumulate = max(round(self.config["TRAIN"]["HYP"]["ACCUMULATE_BATCH_SIZE"] / self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"]), 1)

        for batch_idx, (imgs, targets, paths, _) in enumerate(self.train_dataloader):
            total_batch_idx = batch_idx + (self.train_batches * epoch)
            imgs = imgs.to(self.device).float() / 255.0
            targets = targets.to(self.device)

            # Calculate the time it takes to load a batch of data
            data_time.update(time.time() - end)

            # Plot images with bounding boxes
            visual_anno_path = os.path.join(".", self.config["EXP_NAME"] + ".jpg")
            if total_batch_idx < 1:
                if os.path.exists(visual_anno_path):
                    os.remove(visual_anno_path)
                plot_images(imgs, targets, paths, visual_anno_path, max_size=self.config["TRAIN"]["IMG_SIZE"])

            # Burn-in
            if total_batch_idx <= self.config["TRAIN"]["HYP"]["NUM_BURN"] and self.config["TRAIN"]["OPTIM"]["NAME"] == "SGD":
                xi = [0, self.config["TRAIN"]["HYP"]["NUM_BURN"]]
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    lr_decay = lambda lr: (((1 + math.cos(lr * math.pi / self.config["TRAIN"]["HYP"]["EPOCHS"])) / 2) ** 1.0) * 0.95 + 0.05
                    x["lr"] = np.interp(total_batch_idx, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lr_decay(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(total_batch_idx, xi, [0.9, self.config["TRAIN"]["OPTIM"]["MOMENTUM"]])

            # Multi-Scale
            if self.config["TRAIN"]["MULTI_SCALE"]["ENABLE"]:
                if self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MIN"] % self.config["TRAIN"]["MULTI_SCALE"]["GRID_SIZE"] == 0:
                    raise ValueError("MULTI_SCALE.IMG_SIZE_MIN must be a multiple of MULTI_SCALE.GRID_SIZE")
                if self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MAX"] % self.config["TRAIN"]["MULTI_SCALE"]["GRID_SIZE"] == 0:
                    raise ValueError("MULTI_SCALE.IMG_SIZE_MAX must be a multiple of MULTI_SCALE.GRID_SIZE")

                min_grid = self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MIN"] // self.config["TRAIN"]["MULTI_SCALE"]["GRID_SIZE"]
                max_grid = self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MAX"] // self.config["TRAIN"]["MULTI_SCALE"]["GRID_SIZE"]
                img_size = random.randrange(min_grid, max_grid + 1) * self.config["TRAIN"]["GRID_SIZE"]
                scale_factor = img_size / max(imgs.shape[2:])
                if scale_factor != 1:
                    # new shape (stretched to 32-multiple)
                    new_img_size = [math.ceil(x * scale_factor / self.config["TRAIN"]["MULTI_SCALE"]["GRID_SIZE"]) * self.config["TRAIN"]["GRID_SIZE"]
                                    for x in imgs.shape[2:]]
                    imgs = F_torch.interpolate(imgs, size=new_img_size, mode="bilinear", align_corners=False)

            # Initialize the generator gradient
            self.model.zero_grad(set_to_none=True)

            # Mixed precision training
            with amp.autocast():
                output = self.model(imgs)
                loss, loss_item = compute_loss(output,
                                               targets,
                                               self.model,
                                               self.config["TRAIN"]["IOU_THRESH"],
                                               self.config["TRAIN"]["LOSSES"])
                loss *= self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"] / self.config["TRAIN"]["HYP"]["ACCUMULATE_BATCH_SIZE"]

            # Backpropagation
            self.scaler.scale(loss).backward()

            # update generator weights
            if total_batch_idx % accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # update exponential average models weights
            self.ema_model.update_parameters(self.model)

            # Statistical loss value for terminal data output
            giou_losses.update(loss_item[0], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])
            obj_losses.update(loss_item[1], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])
            cls_losses.update(loss_item[2], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])
            losses.update(loss_item[3], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if (batch_idx + 1) % self.config["TRAIN"]["PRINT_FREQ"] == 0 or (batch_idx + 1) == self.train_batches:
                # Writer Loss to file
                self.tblogger.add_scalar("Train/GIoULoss", loss_item[0], total_batch_idx)
                self.tblogger.add_scalar("Train/ObjLoss", loss_item[1], total_batch_idx)
                self.tblogger.add_scalar("Train/ClsLoss", loss_item[2], total_batch_idx)
                self.tblogger.add_scalar("Train/Loss", loss_item[3], total_batch_idx)

                progress.display(batch_idx + 1)

    def train(self):
        self.load_checkpoint()

        for epoch in range(self.start_epoch, self.config["TRAIN"]["HYP"]["EPOCHS"]):
            self.train_on_epoch(epoch)

            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            mean_p, mean_r, mean_ap, mean_f1 = self.evaler.validate_on_epoch(
                self.model,
                self.test_dataloader,
                self.config["DATASET"]["CLASS_NAMES"],
                self.config["TEST"]["AUGMENT"],
                self.config["TEST"]["CONF_THRESH"],
                self.config["TEST"]["IOU_THRESH"],
                (self.config["TEST"]["IOUV1"], self.config["TEST"]["IOUV2"]),
                self.config["TEST"]["GT_JSON_PATH"],
                self.config["TEST"]["PRED_JSON_PATH"],
                False,
                self.device,
            )

            self.tblogger.add_scalar("Test/Precision", mean_p, epoch)
            self.tblogger.add_scalar("Test/Recall", mean_r, epoch)
            self.tblogger.add_scalar("Test/mAP", mean_ap, epoch)
            self.tblogger.add_scalar("Test/F1", mean_f1, epoch)
            print("\n")

            # Save weights
            self.save_checkpoint(epoch, mean_ap)
