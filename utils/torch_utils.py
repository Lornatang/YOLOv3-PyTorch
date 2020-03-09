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

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models


def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False


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


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
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


def model_info(model, report="summary"):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report is "full":
        print("%5s %40s %9s %12s %20s %10s %10s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma"))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print("%5g %40s %9s %12g %20s %10.3g %10.3g" %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print("Model Summary: %g layers, %g parameters, %g gradients" % (len(list(model.parameters())), n_p, n_g))


def load_classifier(name="resnet101", n=2):
    # Loads a pretrained model reshaped to n-class output

    model = models.__dict__[name](pretrained=True)
    # model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained="imagenet")

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
