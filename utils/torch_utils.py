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
import os
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9998, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        d = self.decay
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()
            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))


def fuse_conv_and_bn(conv, bn):
    # source from https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False


def load_classifier(name="resnet101", n=2):
    # Loads a pretrained model reshaped to n-class output

    model = models.__dict__[name](pretrained=True)

    # Display model properties
    input_size = [3, 224, 224]
    input_space = "RGB"
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for x in [input_size, input_space, input_range, mean, std]:
        print(x + " =", eval(x))

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = torch.nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = torch.nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print("%5s %40s %9s %12s %20s %10s %10s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma"))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print("%5g %40s %9s %12g %20s %10.3g %10.3g" %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print("Model Summary: %g layers, %g parameters, %g gradients" % (len(list(model.parameters())), n_p, n_g))


def scale_image(image, ratio=1.0):  # image(16,3,256,416), ratio=1.0
    # scales a batch of pytorch images while retaining same input shape (cropped or grey-padded)
    height, width = image.shape[2:]
    size = (int(height * ratio), int(width * ratio))  # new size
    p = height - size[0], width - size[1]  # pad/crop pixels
    image = F.interpolate(image, size=size, mode='bilinear', align_corners=False)  # resize
    return F.pad(image, [0, p[1], 0, p[0]], value=0.5) if ratio < 1.0 else image[:, :, :p[0], :p[1]]  # pad/crop


def select_device(device="", apex=False, batch_size=None):
    # device = "cpu" or "cuda:0"
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # if device requested other than "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % gpu_count == 0, f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA " + ("Apex " if apex else "")
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            print(f"{s}\n\t+ device:{i} (name=`{x[i].name}`, total_memory={int(x[i].total_memory / c)}MB)")
    else:
        print("Using CPU")

    print("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
