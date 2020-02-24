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
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
  def __init__(self, log_dir):
    r""" Creates a `SummaryWriter` that will write out events and summaries
    to `log_dir`.

    Args:
      log_dir (string): Save directory location. Default is
        runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
        Use hierarchical folder structure to compare
        between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
        for each new experiment to compare across them.

    Examples:
      >>> from torch.utils.tensorboard import SummaryWriter
      # create a summary writer with automatically generated folder name.
      >>> writer = SummaryWriter()
      # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

      # create a summary writer using the specified folder name.
      >>> writer = SummaryWriter("logs")
      # folder location: logs

      # create a summary writer with comment appended.
      >>> writer = SummaryWriter(comment="acc1")
      # folder location: runs/Feb22_10-20-54_s-MacBook-Pro.acc1/
    """

    self.writer = SummaryWriter(log_dir=log_dir)

  def scalar_summary(self, tag, scalar_value, global_step=None):
    r""" Add scalar data to summary.

    Args:
      tag (string): Data identifier.
      scalar_value (float or string/blobname): Value to save.
      global_step (int): Global step value to record.

    Examples:
      >>> from torch.utils.tensorboard import SummaryWriter
      >>> writer = SummaryWriter()
      >>> x = range(100)
      >>> for i in x:
      >>>   writer.add_scalar('y=2x', i * 2, i)
      >>> writer.close()
    """

    self.writer.add_scalar(tag, scalar_value, global_step)

  def list_of_scalars_summary(self, tag_and_value, step):
    r""" Add scalar data to summary."""
    for tag, value in tag_and_value:
      self.writer.add_scalar(tag, value, step)
