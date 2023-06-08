# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess

import logging

from aie_runtime.bean import ConvertConfig


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Convert:
    def __init__(self,
        config: ConvertConfig
    ) -> None:
        self._config = config
        self.python_version = sys.executable or "python3"

    @classmethod
    def execute_command(cls, cmd):
        """
        Function Description:
            run the following command
        Parameter:
            cmd: command
        Return Value:
            command output result
        Exception Description:
            when invalid command throw exception
        """
        logger.info('Execute command:%s' % cmd)
        process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while process.poll() is None:
            line = process.stdout.readline()
            line = line.strip()
            if line:
                logger.debug(line)
        if process.returncode != 0:
            logger.error('Failed to execute command:%s' % " ".join(cmd))

    def convert_model(self) -> None:
        cur_dir = os.path.dirname(__file__)
        ait_convert = os.path.join(cur_dir, "../ait_convert")
        run_cmd = [ait_convert, self._config.model, self._config.output, self._config.soc_version]
        self.execute_command(run_cmd)
        logger.info("AIE model convert finished, the command: %s", run_cmd)
