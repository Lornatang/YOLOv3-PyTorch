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
import torch.nn as nn

from model.module.activition import Mish
from model.module.activition import Swish


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 batch_norm=False, activation=None):
        super(BasicConv2d, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, bias=not batch_norm)]

        if batch_norm:
            layers.append(
                nn.BatchNorm2d(out_channels, momentum=0.003, eps=0.0001))

        if activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "swish":
            layers.append(Swish())
        elif activation == "mish":
            layers.append(Mish())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)

        return out


class DeepConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeepConv2d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, momentum=0.003, eps=0.0001),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, padding=0, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.003, eps=0.0001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.main(x)

        return out
