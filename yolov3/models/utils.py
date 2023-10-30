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
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Divisor to the number of channels.

    Args:
        v (float): input value
        divisor (int): divisor
        min_value (int): minimum value

    Returns:
        int: divisible value
    """

    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


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
    if state_dict is None:
        raise ValueError(f"state_dict is None")

    # Traverse the model parameters and load the parameters in the pre-trained model into the current model
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in state_dict.keys() and v.size() == state_dict[k].size()}

    # update model parameters
    state_dict.update(new_state_dict)
    model.load_state_dict(state_dict)

    return model


def load_resume_state_dict(
        model: nn.Module,
        model_weights_path: str | Path,
        ema_model: nn.Module or None,
        optimizer: optim.Optimizer,
) -> tuple[nn.Module, nn.Module, int, float, optim.Optimizer]:
    """Load the PyTorch model weights from the model weight address

    Args:
        model (nn.Module): PyTorch model
        model_weights_path: PyTorch model path
        ema_model (nn.Module): EMA model
        optimizer (optim.Optimizer): Optimizer

    Returns:
        model: PyTorch model with weights
        model_weights_path: PyTorch model path
        ema_model (nn.Module): EMA model
        start_epoch (int): Start epoch
        best_map50 (float): Best map50
        optimizer (optim.Optimizer): Optimizer
    """

    model_weights_path = model_weights_path if isinstance(model_weights_path, str) else str(model_weights_path)

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weight file not found '{model_weights_path}'")

    if model_weights_path.endswith(".weights"):
        raise ValueError(f"You loaded darknet model weights '{model_weights_path}', must be converted to PyTorch model weights")

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint["epoch"] if checkpoint["epoch"] else 0
    best_map50 = checkpoint["best_map50"] if checkpoint["best_map50"] else 0.0

    model = load_state_dict(model, checkpoint["state_dict"])
    if checkpoint["ema_state_dict"] is not None:
        ema_model = load_state_dict(ema_model, checkpoint["ema_state_dict"])

    optimizer_state_dict = checkpoint["optimizer"]
    if optimizer_state_dict is None:
        raise ValueError(f"Model weight file '{model_weights_path}' not have 'optimizer'")
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, ema_model, start_epoch, best_map50, optimizer
