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

from .activition import HSigmoid
from .activition import HSwish


class MobileNetv3_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MobileNetv3_Conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            HSwish(),
        )

    def forward(self, x):
        out = self.main(x)
        return out


class SeModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
