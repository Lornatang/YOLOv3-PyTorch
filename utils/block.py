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


class MobileNetv3_Block(nn.Module):
    """expand + depth_wise + point_wise"""

    def __init__(self, kernel_size, in_features, expand_size, out_features,
                 activation, semodule, stride):
        super(MobileNetv3_Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_features, expand_size, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = activation
        self.conv2 = nn.Conv2d(expand_size, expand_size,
                               kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = activation
        self.conv3 = nn.Conv2d(expand_size, out_features, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_features),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
