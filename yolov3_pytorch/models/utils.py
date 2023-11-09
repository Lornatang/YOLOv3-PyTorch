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
import os
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn, optim

from .darknet import Darknet
import warnings

__all__ = [
    "convert_model_state_dict", "load_state_dict", "load_resume_state_dict", "load_darknet_weights", "save_darknet_weights",
]


def convert_model_state_dict(model_config_path: Union[str, Path], model_weights_path: Union[str, Path]) -> None:
    """

    Args:
        model_config_path (str or Path): Model configuration file path.
        model_weights_path (str or Path): path to darknet models weights file
    """

    # Initialize models
    model = Darknet(model_config_path)

    # Load weights and save
    # if PyTorch format
    if model_weights_path.endswith(".pth.tar"):
        state_dict = torch.load(model_weights_path, map_location="cpu")["state_dict"]
        model = load_state_dict(model, state_dict)

        target = model_weights_path[:-8] + ".weights"
        save_darknet_weights(model, target)
    # Darknet format
    elif model_weights_path.endswith(".weights"):
        model = load_darknet_weights(model, model_weights_path)

        chkpt = {"epoch": 0,
                 "best_mean_ap": 0.0,
                 "state_dict": model.state_dict(),
                 "ema_state_dict": None}

        target = model_weights_path[:-8] + ".pth.tar"
        torch.save(chkpt, target)
    else:
        raise ValueError(f"Model weight file '{model_weights_path}' not supported. Only support '.pth.tar' and '.weights'")
    print(f"Success: converted '{model_weights_path}' to '{target}'")


def load_state_dict(
        model: nn.Module,
        state_dict: dict,
) -> nn.Module:
    """Load the PyTorch model weights from the model weight address

    Args:
        model (nn.Module): PyTorch model
        state_dict: PyTorch model state dict

    Returns:
        model: PyTorch model with weights
    """

    # When the PyTorch version is less than 2.0, the model compilation is not supported.
    if int(torch.__version__[0]) < 2 and model.compile_mode:
        warnings.warn("PyTorch version is less than 2.0, does not support model compilation.")
        model.compile_mode = False

    # compile keyword
    compile_keyword = ""
    if model.compile_mode:
        compile_keyword = "_orig_mod"

    # Create new OrderedDict that does not contain the module prefix
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    # Remove the module prefix and update the model weight
    for k, v in state_dict.items():
        k_prefix = k.split(".")[0]

        if k_prefix == compile_keyword and not model.compile_mode:
            name = k[len(compile_keyword) + 1:]
        elif k_prefix != compile_keyword and model.compile_mode:
            raise ValueError("The model is not compiled, but the weight is compiled.")
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Filter out unnecessary parameters
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # Update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_resume_state_dict(
        model: nn.Module,
        ema_model: nn.Module,
        model_weights_path: str | Path,
) -> tuple[int, float, nn.Module, nn.Module, optim.Optimizer]:
    """Load the PyTorch model weights from the model weight address

    Args:
        model (nn.Module): PyTorch model
        model_weights_path: PyTorch model path
        ema_model (nn.Module): EMA model

    Returns:
        start_epoch (int): Start epoch
        best_mean_ap (float): Best mean ap
        model: PyTorch model with weights
        model_weights_path: PyTorch model path
        ema_model (nn.Module): EMA model
        optimizer (optim.Optimizer): Optimizer
    """

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found '{model_weights_path}'")

    if model_weights_path.endswith(".weights"):
        raise ValueError(f"You loaded darknet model weights '{model_weights_path}', must be converted to PyTorch model weights")

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint["epoch"]
    best_mean_ap = checkpoint["best_mean_ap"]
    model = load_state_dict(model, checkpoint["state_dict"])
    ema_model = load_state_dict(ema_model, checkpoint["ema_state_dict"])
    optimizer = checkpoint["optimizer"]

    return start_epoch, best_mean_ap, model, ema_model, optimizer


def load_darknet_weights(model: nn.Module, weights_path: Union[str, Path]) -> Darknet:
    r"""Parses and loads the weights stored in 'weights_path'

    Args:
        model (Darknet): models to load the weights into.
        weights_path (Union[str, Path]): Path to the weights file.
    """

    # Open the weights file
    with open(weights_path, "rb") as f:
        # First five are header values
        header = np.fromfile(f, dtype=np.int32, count=5)
        model.header_info = header  # Needed to write header when saving weights
        model.seen = header[3]  # number of imgs seen during training
        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    # Establish cutoff for loading backbone weights
    cutoff = None
    # If the weights file has a cutoff, we can find out about it by looking at the filename
    # examples: darknet53.conv.74 -> cutoff is 74
    filename = os.path.basename(weights_path)
    if ".conv." in filename:
        try:
            cutoff = int(filename.split(".")[-1])  # use last part of filename
        except ValueError:
            pass

    ptr = 0
    for i, (module_define, module) in enumerate(zip(model.module_defines, model.module_list)):
        if i == cutoff:
            break
        if module_define["type"] == "convolutional":
            conv_layer = module[0]
            if module_define["batch_normalize"]:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return model


def save_darknet_weights(model: nn.Module, weights_path: Union[str, Path], cutoff: int = -1) -> None:
    r"""Saves the models in darknet format.

    Args:
        model (Darknet): models to save.
        weights_path: path of the new weights file
        cutoff: save layers between 0 and cutoff (cutoff = -1 -> all are saved) Default: -1
    """

    fp = open(weights_path, "wb")
    model.header_info[3] = model.seen
    model.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_define, module) in enumerate(zip(model.module_defines[:cutoff], model.module_list[:cutoff])):
        if module_define["type"] == "convolutional":
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_define["batch_normalize"]:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()
