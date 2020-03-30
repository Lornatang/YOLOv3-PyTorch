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

ONNX_EXPORT = False


def create_grids(self, image_size=416, ng=(13, 13), device="cpu",
                 dtype=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(image_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(dtype).view(
        (1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).type(dtype)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    parameter_num = sum(x.numel() for x in model.parameters())
    gradient_num = sum(x.numel() for x in model.parameters() if x.requires_grad)

    print(f"Model Summary: {len(list(model.parameters()))} layers, "
          f"{parameter_num} parameters, {gradient_num} gradients")
