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


def parse_model_configure(path):
  r""" Parses the yolo-v3 layer configuration file and returns module definitions

  Args:
    path (str): Correct yolo v3 profile address.

  Examples:
    >>> model_layer = parse_model_configure("coco.cfg")
    [{"type": "net",
    "batch": "16",
    "subdivisions": "1",
    "width": "416",
    "height": "416",
    "channels": "3",
    "momentum": "0.9",
    "decay": "0.0005",
    "angle": "0",
    "saturation": "1.5",
    "exposure": "1.5",
    "hue": ".1",
    "learning_rate": "0.001",
    "burn_in": "1000",
    "max_batches": "500200",
    "policy": "steps",
    "steps": "400000,450000",
    "scales": ".1,.1"},
    ...
    ]

    Returns:
      Dictionary with parameters.
  """

  file = open(path, "r")
  lines = file.read().split("\n")
  lines = [x for x in lines if x and not x.startswith("#")]  # do not read comments
  lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

  module_defs = []

  for line in lines:
    if line.startswith("["):  # This marks the start of a new block
      module_defs.append({})
      module_defs[-1]["type"] = line[1:-1].rstrip()
      if module_defs[-1]["type"] == "convolutional":
        module_defs[-1]["batch_normalize"] = 0
    else:
      key, value = line.split("=")
      value = value.strip()
      module_defs[-1][key.rstrip()] = value.strip()

  return module_defs


def parse_data_configure(path):
  r""" Parses the data configuration file

  Args:
    path (str): Path to data config file.

  Examples:
    >>> data_configure = parse_data_configure("coco2014.data")
    {'gpus': '0,1,2,3',
     'num_workers': '16',
     'classes': '80',
     'train': 'data/coco2014/train.txt',
     'valid': 'data/coco2014/valid.txt',
     'names': 'data/coco2014.names',
     'backup': 'backup/',
     'eval': 'coco2014'}

  Returns:
    A string of dictionaries with data parameters
  """

  parameter = dict()
  parameter["gpus"] = "0,1,2,3"  # default use four GPUs
  parameter["num_workers"] = "16"  # every GPU load four thread data

  with open(path, "r") as f:
    lines = f.readlines()

  for line in lines:
    line = line.strip()
    if line == "" or line.startswith("#"):  # do not read comments and last white line
      continue
    key, value = line.split("=")
    parameter[key.strip()] = value.strip()
  return parameter
