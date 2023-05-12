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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import logging.handlers

IS_PYTHON3 = sys.version_info > (3,)
logger = logging.getLogger('porting_advisor')

if os.path.exists("porting_advisor.log"):
    os.remove("porting_advisor.log")

# create console handler and formatter for logger
console = logging.StreamHandler()
fh = logging.handlers.TimedRotatingFileHandler("porting_advisor.log", when='midnight', interval=1, backupCount=7)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[%(lineno)d] - %(message)s")

console.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(fh)
