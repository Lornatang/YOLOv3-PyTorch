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


def parse_dataset_config(path: str) -> dict:
    """Parses the data configuration file

    Args:
        path (str): path to data config file

    Returns:
        data_config (dict): A dictionary containing the information from the data config file

    """
    if not os.path.exists(path) and os.path.exists("data" + os.sep + path):  # add data/ prefix if omitted
        path = "data" + os.sep + path

    with open(path, "r") as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        key, val = line.split("=")
        options[key.strip()] = val.strip()

    return options
