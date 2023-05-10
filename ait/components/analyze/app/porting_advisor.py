# Copyright 2023 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.log_util import logger
from porting.app import init_args, start_scan_kit


def start_analyze():
    args = init_args()
    logger.setLevel(args.log_level)
    start_scan_kit(args)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    start_analyze()
