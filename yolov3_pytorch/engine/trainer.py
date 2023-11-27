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
import time
from pathlib import Path
from typing import Dict, Union

import torch
import torch.utils.data
from torch import optim
from torch.cuda import amp
from torch.nn import functional as F_torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter
from yolov3_pytorch.data import BaseDatasets
from yolov3_pytorch.engine.evaler import Evaler
from yolov3_pytorch.models import Darknet
from yolov3_pytorch.models.losses import compute_loss
from yolov3_pytorch.models.utils import load_state_dict
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
            save_weights_dir: Union[str, Path],
            tblogger: SummaryWriter,
    ) -> None:
        self.config = config
        self.scaler = scaler
        self.device = device
        self.save_weights_dir = save_weights_dir
        self.tblogger = tblogger

        self.train_datasets, self.val_datasets, self.train_dataloader, self.val_dataloader = self.load_datasets()
        self.train_batches = len(self.train_dataloader)
        self.model, self.ema_model = self.build_model()
        self.optim = self.define_optim()
        self.lr_scheduler = self.define_lr_scheduler()

        self.start_epoch = 0
        self.best_mean_ap = 0.0
        self.eval = Evaler(self.config, self.device)

    def load_datasets(self) -> tuple:
        r"""Load training and test datasets from a configuration file, such as yaml

        Returns:
            tuple: train_dataloader, test_dataloader

        """
        if self.config["TRAIN"]["DATASET"]["SINGLE_CLASSES"]:
            self.config["MODEL"]["NUM_CLASSES"] = 1
        self.config["TRAIN"]["LOSSES"]["CLS_LOSS"]["WEIGHT"] *= self.config["MODEL"]["NUM_CLASSES"] / 80

        train_datasets = BaseDatasets(self.config["TRAIN"]["DATASET"]["ROOT"],
                                      self.config["MODEL"]["IMG_SIZE"],
                                      self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                      self.config["TRAIN"]["DATASET"]["AUGMENT"],
                                      self.config["AUGMENT"]["HYP"],
                                      self.config["TRAIN"]["DATASET"]["RECT_LABEL"],
                                      self.config["TRAIN"]["DATASET"]["CACHE_IMAGES"],
                                      self.config["TRAIN"]["DATASET"]["SINGLE_CLASSES"],
                                      pad=0.0,
                                      gray=self.config["MODEL"]["GRAY"])
        val_datasets = BaseDatasets(self.config["VAL"]["DATASET"]["ROOT"],
                                    self.config["MODEL"]["IMG_SIZE"],
                                    self.config["VAL"]["HYP"]["IMGS_PER_BATCH"],
                                    self.config["VAL"]["DATASET"]["AUGMENT"],
                                    self.config["AUGMENT"]["HYP"],
                                    self.config["VAL"]["DATASET"]["RECT_LABEL"],
                                    self.config["VAL"]["DATASET"]["CACHE_IMAGES"],
                                    self.config["VAL"]["DATASET"]["SINGLE_CLASSES"],
                                    pad=0.5,
                                    gray=self.config["MODEL"]["GRAY"])
        train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                       batch_size=self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                                       shuffle=True,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       drop_last=True,
                                                       persistent_workers=True,
                                                       collate_fn=train_datasets.collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_datasets,
                                                     batch_size=self.config["VAL"]["HYP"]["IMGS_PER_BATCH"],
                                                     shuffle=False,
                                                     num_workers=4,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True,
                                                     collate_fn=val_datasets.collate_fn)

        return train_datasets, val_datasets, train_dataloader, val_dataloader

    def build_model(self) -> tuple:
        # Create model
        model = Darknet(self.config["MODEL"]["CONFIG_PATH"],
                        self.config["MODEL"]["IMG_SIZE"],
                        self.config["MODEL"]["GRAY"],
                        self.config["MODEL"]["COMPILED"])
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

    def define_optim(self) -> optim:
        if self.config["TRAIN"]["OPTIM"]["NAME"] == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   self.config["TRAIN"]["OPTIM"]["LR"],
                                   (self.config["TRAIN"]["OPTIM"]["BETA1"], self.config["TRAIN"]["OPTIM"]["BETA2"]),
                                   weight_decay=self.config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
        elif self.config["TRAIN"]["OPTIM"]["NAME"] == "sgd":
            optimizer = optim.SGD(self.model.parameters(),
                                  self.config["TRAIN"]["OPTIM"]["LR"],
                                  self.config["TRAIN"]["OPTIM"]["MOMENTUM"],
                                  weight_decay=self.config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"],
                                  nesterov=self.config["TRAIN"]["OPTIM"]["NESTEROV"])

        else:
            raise NotImplementedError("Only Support ['sgd', 'adam'] optimizer")

        return optimizer

    def define_lr_scheduler(self) -> optim.lr_scheduler:
        r"""Define a learning rate scheduler

        Returns:
            torch.optim.lr_scheduler: learning rate scheduler
        """
        lr_scheduler_name = self.config["TRAIN"]["OPTIM"]["LR_SCHEDULER"]["NAME"]
        if lr_scheduler_name == "step_lr":
            scheduler = optim.lr_scheduler.StepLR(self.optim,
                                                  self.config["TRAIN"]["OPTIM"]["LR_SCHEDULER"]["STEP_SIZE"],
                                                  self.config["TRAIN"]["OPTIM"]["LR_SCHEDULER"]["GAMMA"])
        elif lr_scheduler_name == "cosine_with_warm":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, self.config["TRAIN"]["OPTIM"]["LR_SCHEDULER"]["T_0"])
        else:
            raise NotImplementedError("Only Support ['step_lr', 'cosine_with_warm'] lr_scheduler")
        return scheduler

    def train_on_epoch(self, epoch):
        # The information printed by the progress bar
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        iou_losses = AverageMeter("IoULoss", ":.4e")
        obj_losses = AverageMeter("ObjLoss", ":.4e")
        cls_losses = AverageMeter("ClsLoss", ":.4e")
        progress = ProgressMeter(self.train_batches,
                                 [batch_time, data_time, iou_losses, obj_losses, cls_losses],
                                 prefix=f"Epoch: [{epoch}]")

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
            visual_anno_path = os.path.join("./results/train", self.config["EXP_NAME"] + ".jpg")
            if total_batch_idx < 1:
                if os.path.exists(visual_anno_path):
                    os.remove(visual_anno_path)
                plot_images(imgs, targets, paths, visual_anno_path, max_size=self.config["MODEL"]["IMG_SIZE"])

            # Multi-Scale
            if self.config["TRAIN"]["MULTI_SCALE"]["ENABLE"]:
                if self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MIN"] % self.config["MODEL"]["GRID_SIZE"] != 0:
                    raise ValueError("MULTI_SCALE.IMG_SIZE_MIN must be a multiple of MODEL.GRID_SIZE")
                if self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MAX"] % self.config["MODEL"]["GRID_SIZE"] != 0:
                    raise ValueError("MULTI_SCALE.IMG_SIZE_MAX must be a multiple of MODEL.GRID_SIZE")

                min_grid = self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MIN"] // self.config["MODEL"]["GRID_SIZE"]
                max_grid = self.config["TRAIN"]["MULTI_SCALE"]["IMG_SIZE_MAX"] // self.config["MODEL"]["GRID_SIZE"]
                img_size = random.randrange(min_grid, max_grid + 1) * self.config["MODEL"]["GRID_SIZE"]
                scale_factor = img_size / max(imgs.shape[2:])
                if scale_factor != 1:
                    # new shape (stretched to 32-multiple)
                    new_img_size = [math.ceil(x * scale_factor / self.config["MODEL"]["GRID_SIZE"]) * self.config["MODEL"]["GRID_SIZE"]
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
            # Scale loss
            loss *= self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"] / self.config["TRAIN"]["HYP"]["ACCUMULATE_BATCH_SIZE"]
            # Backpropagation
            self.scaler.scale(loss).backward()

            # update generator weights
            if (total_batch_idx + 1) % accumulate == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                # update exponential average models weights
                self.ema_model.update_parameters(self.model)

            # Statistical loss value for terminal data output
            iou_losses.update(loss_item[0], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])
            obj_losses.update(loss_item[1], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])
            cls_losses.update(loss_item[2], self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_idx % self.config["TRAIN"]["PRINT_FREQ"] == 0 or (batch_idx + 1) % self.train_batches == 0:
                # Writer Loss to file
                self.tblogger.add_scalar("Train/IoULoss", loss_item[0], total_batch_idx)
                self.tblogger.add_scalar("Train/ObjLoss", loss_item[1], total_batch_idx)
                self.tblogger.add_scalar("Train/ClsLoss", loss_item[2], total_batch_idx)

                progress.display(batch_idx + 1)

    def train(self):
        self.load_checkpoint()

        for epoch in range(self.start_epoch, self.config["TRAIN"]["HYP"]["EPOCHS"]):
            self.train_on_epoch(epoch)

            mean_p, mean_r, mean_ap, mean_f1 = self.eval.validate_on_epoch(
                self.model,
                self.val_dataloader,
                self.config["CLASS_NAMES"],
                self.config["VAL"]["DATASET"]["AUGMENT"],
                self.config["VAL"]["CONF_THRESH"],
                self.config["VAL"]["IOU_THRESH"],
                eval(self.config["VAL"]["IOUV"]),
                self.config["VAL"]["GT_JSON_PATH"],
                self.config["VAL"]["PRED_JSON_PATH"],
                self.config["VAL"]["VERBOSE"],
                self.device,
            )
            self.tblogger.add_scalar("Val/Precision", mean_p, epoch)
            self.tblogger.add_scalar("Val/Recall", mean_r, epoch)
            self.tblogger.add_scalar("Val/mAP", mean_ap, epoch)
            self.tblogger.add_scalar("Val/F1", mean_f1, epoch)
            print("\n")

            # Update learning rate scheduler
            self.lr_scheduler.step()

            # Save weights
            self.save_checkpoint(epoch, mean_ap)

    def load_checkpoint(self) -> None:
        def _load(weights_path: str) -> None:
            if os.path.isfile(weights_path):
                with open(weights_path, "rb") as f:
                    checkpoint = torch.load(f, map_location=self.device)
                self.start_epoch = checkpoint.get("epoch", 0)
                self.best_mean_ap = checkpoint.get("best_mean_ap", 0.0)
                load_state_dict(self.model, checkpoint.get("state_dict", {}))
                load_state_dict(self.ema_model, checkpoint.get("ema_state_dict", {}))
                load_state_dict(self.optim, checkpoint.get("optim_state_dict", {}))
                load_state_dict(self.lr_scheduler, checkpoint.get("lr_scheduler_state_dict", {}))
                print(f"Loaded checkpoint '{weights_path}'")
            else:
                raise FileNotFoundError(f"No checkpoint found at '{weights_path}'")

        pretrained_weights = self.config["TRAIN"]["CHECKPOINT"]["PRETRAINED_WEIGHTS"]
        resume_weights = self.config["TRAIN"]["CHECKPOINT"]["RESUME_WEIGHTS"]
        if pretrained_weights:
            _load(pretrained_weights)
        elif resume_weights:
            _load(resume_weights)
        else:
            print("No checkpoint or pretrained weights found, train from scratch")

    def save_checkpoint(self, epoch: int, mean_ap: float) -> None:
        # Automatically save models weights
        is_best = mean_ap > self.best_mean_ap
        self.best_mean_ap = max(mean_ap, self.best_mean_ap)

        state_dict = {
            "epoch": epoch + 1,
            "best_mean_ap": self.best_mean_ap,
            "state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema_model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
        }

        if (epoch + 1) % self.config["TRAIN"]["SAVE_EVERY_EPOCH"] == 0:
            weights_path = os.path.join(self.save_weights_dir, f"epoch_{epoch:06d}.pth.tar")
            torch.save(state_dict, weights_path)

        if is_best:
            weights_path = os.path.join(self.save_weights_dir, "best.pth.tar")
            torch.save(state_dict, weights_path)

        weights_path = os.path.join(self.save_weights_dir, "last.pth.tar")
        torch.save(state_dict, weights_path)
