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


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, ch1x1, out_channels,
                 activation="None"):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            BasicConv2d(in_channels, ch1x1, 1, 1, 0,
                        batch_norm=True, activation=activation),
            BasicConv2d(ch1x1, out_channels, 3, 1, 1,
                        batch_norm=True, activation=activation)
        )

    def forward(self, x):
        shortcut = x
        out = self.main(x)
        out += shortcut

        return out
