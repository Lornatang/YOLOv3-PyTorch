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
import torch
import torch.nn as nn

from ..module import BasicConv2d


class Darknet19(torch.nn.Module):
    """ Some improvements have been made to yolo1, please refer to https://arxiv.org/abs/1612.08242
    """

    def __init__(self):
        super(Darknet19, self).__init__()
        self.conv1 = BasicConv2d(3, 32, 3, 1, 1,
                                 batch_norm=True, activation="leakyrelu")

        self.conv2 = BasicConv2d(32, 64, 3, 1, 1,
                                 batch_norm=True, activation="leakyrelu")

        self.conv3_1 = BasicConv2d(64, 128, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv3_2 = BasicConv2d(128, 64, 1, 1, 0,
                                   batch_norm=True, activation="leakyrelu")
        self.conv3_3 = BasicConv2d(64, 128, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")

        self.conv4_1 = BasicConv2d(128, 256, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv4_2 = BasicConv2d(256, 128, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv4_3 = BasicConv2d(128, 256, 1, 1, 0,
                                   batch_norm=True, activation="leakyrelu")

        self.conv5_1 = BasicConv2d(256, 512, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv5_2 = BasicConv2d(512, 256, 1, 1, 0,
                                   batch_norm=True, activation="leakyrelu")
        self.conv5_3 = BasicConv2d(256, 512, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv5_4 = BasicConv2d(512, 256, 1, 1, 0,
                                   batch_norm=True, activation="leakyrelu")
        self.conv5_5 = BasicConv2d(256, 512, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")

        self.conv6_1 = BasicConv2d(256, 512, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv6_2 = BasicConv2d(512, 256, 1, 1, 0,
                                   batch_norm=True, activation="leakyrelu")
        self.conv6_3 = BasicConv2d(256, 512, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")
        self.conv6_4 = BasicConv2d(512, 256, 1, 1, 0,
                                   batch_norm=True, activation="leakyrelu")
        self.conv6_5 = BasicConv2d(256, 512, 3, 1, 1,
                                   batch_norm=True, activation="leakyrelu")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        small_output = self.maxpool(x)  # 256 * 28 * 28

        x = self.conv5_1(small_output)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        medium_output = self.maxpool(x)  # 512 * 14 * 14

        x = self.conv6_1(medium_output)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.conv6_4(x)
        large_output = self.conv6_5(x)  # 1024 * 7 * 7

        return small_output, medium_output, large_output
