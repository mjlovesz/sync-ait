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
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bean import ConvertConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Convert:
    def __init__(self,
        config: ConvertConfig
    ) -> None:
        self._config = config
        self.python_version = sys.executable or "python3"


    def convert_model(self) -> None:
        retval = self.aie_build_model()
        self.aie_model_convert(retval)
        os.chdir(retval)


    def aie_build_model(self) -> None:
        execute_path = "./components/convert/aie_runtime/cpp"
        convert_sh_cmd = ["sh", "build.sh", "-p", self.python_version]
        retval = os.getcwd()
        os.chdir(execute_path)
        self.execute_command(convert_sh_cmd)
        logger.info("Run command line: %s" % (convert_sh_cmd))
        return retval


    def aie_model_convert(self, retval) -> None:
        run_path = "./build"
        os.chdir(run_path)
        run_cmd = ["./ait_convert", self._config.model, self._config.output, self._config.soc_version]
        self.execute_command(run_cmd)

        curval = os.getcwd()
        output_model = self._config.output
        copy_cmd = ["cp", output_model, retval]
        self.execute_command(copy_cmd)
        logger.info("AIE model convert finished, the command: %s" % (run_cmd))


    @classmethod
    def execute_command(self, cmd):
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
                logger.info(line)
        if process.returncode != 0:
            logger.error('Failed to execute command:%s' % " ".join(cmd))



