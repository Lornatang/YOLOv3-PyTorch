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
"""
PyTorch model weight and DarkNet interchange
"""
import argparse

from yolov3_pytorch.models.utils import convert_model_state_dict


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch model weight and DarkNet interchange")
    parser.add_argument("--cfg", type=str, required=True, help="input model configuration file")
    parser.add_argument("--weights", type=str, required=True, help="input model weight file")

    opts = parser.parse_args()

    return opts


if __name__ == "__main__":
    opts = get_opts()

    convert_model_state_dict(opts.cfg, opts.weights)
