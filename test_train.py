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
from model.network.yolov3_voc import VOC
from utils.loss import YoloV3Loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.voc_dataset as data
import random
import argparse
from utils import select_device
from utils import init_seeds

DATA_PATH = "/home/unix/dataset/VOC"
PROJECT_PATH = "/home/unix/code/One-Stage-Detector/yolo"

DATA = {"CLASSES": ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor'],
        "NUM": 20}

# model
MODEL = {"ANCHORS": [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],
                     # Anchors for small obj
                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],
                     # Anchors for medium obj
                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],
         # Anchors for big obj
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3
         }

# train
TRAIN = {
    "TRAIN_IMG_SIZE": 448,
    "AUGMENT": True,
    "BATCH_SIZE": 8,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 50,
    "NUMBER_WORKERS": 4,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2  # or None
}

# test
TEST = {
    "TEST_IMG_SIZE": 544,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": False,
    "FLIP_TEST": False
}


class Trainer(object):
    def __init__(self):
        init_seeds(0)
        self.device = select_device(apex=True)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = TRAIN["EPOCHS"]
        self.multi_scale_train = TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train",
                                             img_size=TRAIN[
                                                 "TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=TRAIN["BATCH_SIZE"],
                                           num_workers=TRAIN["NUMBER_WORKERS"],
                                           shuffle=True)
        self.yolov3 = VOC().to(self.device)

        self.optimizer = optim.SGD(self.yolov3.parameters(),
                                   lr=TRAIN["LR_INIT"],
                                   momentum=TRAIN["MOMENTUM"],
                                   weight_decay=TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV3Loss(anchors=MODEL["ANCHORS"],
                                    strides=MODEL["STRIDES"],
                                    iou_threshold_loss=TRAIN[
                                        "IOU_THRESHOLD_LOSS"])

    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (
                    imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes,
                    mbboxes,
                    lbboxes) in enumerate(self.train_dataloader):

                self.optimizer.step()

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d,
                                                                      label_sbbox,
                                                                      label_mbbox,
                                                                      label_lbbox,
                                                                      sbboxes,
                                                                      mbboxes,
                                                                      lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor(
                    [loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:
                    s = (
                            'Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                            'lr: %g') % (epoch, self.epochs - 1, i,
                                         len(self.train_dataloader) - 1,
                                         mloss[0], mloss[1], mloss[2], mloss[3],
                                         self.optimizer.param_groups[0]['lr'])
                    print(s)

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i + 1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(
                        range(10, 20)) * 32
                    print("multi_scale_img_size : {}".format(
                        self.train_dataset.img_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    Trainer().train()
