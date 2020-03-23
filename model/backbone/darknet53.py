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

from model import BasicConv2d
from model import ResidualBlock


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = BasicConv2d(3, 32, 3, 1, 1,
                                 batch_norm=True, activation='leakyrelu')

        self.conv2 = BasicConv2d(32, 64, 3, 2, 1,
                                 batch_norm=True, activation='leakyrelu')

        self.rb1 = ResidualBlock(in_channels=64, ch1x1=32, out_channels=64)

        self.conv3 = BasicConv2d(64, 128, 3, 2, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.rb2_1 = ResidualBlock(in_channels=128, ch1x1=64, out_channels=128)
        self.rb2_2 = ResidualBlock(in_channels=128, ch1x1=64, out_channels=128)

        self.conv4 = BasicConv2d(128, 256, 3, 2, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.rb3_1 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_2 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_3 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_4 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_5 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_6 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_7 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)
        self.rb3_8 = ResidualBlock(in_channels=256, ch1x1=128, out_channels=256)

        self.conv5 = BasicConv2d(256, 512, 3, 2, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.rb4_1 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_2 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_3 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_4 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_5 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_6 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_7 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)
        self.rb4_8 = ResidualBlock(in_channels=512, ch1x1=256, out_channels=512)

        self.conv6 = BasicConv2d(512, 1024, 3, 2, 1,
                                 batch_norm=True, activation='leakyrelu')
        self.rb5_1 = ResidualBlock(in_channels=1024, ch1x1=512, out_channels=1024)
        self.rb5_2 = ResidualBlock(in_channels=1024, ch1x1=512, out_channels=1024)
        self.rb5_3 = ResidualBlock(in_channels=1024, ch1x1=512, out_channels=1024)
        self.rb5_4 = ResidualBlock(in_channels=1024, ch1x1=512, out_channels=1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.rb1(x)

        x = self.conv3(x)
        x = self.rb2_1(x)
        x = self.rb2_2(x)

        x = self.conv4(x)
        x = self.rb3_1(x)
        x = self.rb3_2(x)
        x = self.rb3_3(x)
        x = self.rb3_4(x)
        x = self.rb3_5(x)
        x = self.rb3_6(x)
        x = self.rb3_7(x)
        small_output = self.rb3_8(x)  # small

        x = self.conv5(small_output)
        x = self.rb4_1(x)
        x = self.rb4_2(x)
        x = self.rb4_3(x)
        x = self.rb4_4(x)
        x = self.rb4_5(x)
        x = self.rb4_6(x)
        x = self.rb4_7(x)
        medium_output = self.rb4_8(x)  # medium

        x = self.conv6(medium_output)
        x = self.rb5_1(x)
        x = self.rb5_2(x)
        x = self.rb5_3(x)
        large_output = self.rb5_4(x)  # large

        return small_output, medium_output, large_output
