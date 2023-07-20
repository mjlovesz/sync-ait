#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023-2023 Huawei Technologies Co., Ltd.
#
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
"""
Function:
This class mainly involves convert model to json function.
"""
import os

from msquickcmp.common import utils
from msquickcmp.common.utils import AccuracyCompareException

ATC_FILE_PATH = "compiler/bin/atc"
OLD_ATC_FILE_PATH = "atc/bin/atc"


class AtcUtils(object):
    """
    Class for convert the om model to json.
    """

    def __init__(self, arguments):
        self.arguments = arguments

    def convert_model_to_json(self):
        """
        Function Description:
            convert om model to json
        Return Value:
            output json path
        Exception Description:
            when the model type is wrong throw exception
        """
        model_name, extension = utils.get_model_name_and_extension(self.arguments.offline_model_path)
        if ".om" != extension:
            utils.logger.error('The offline model file ends with an .om file.Please check {} file.'.format(
                self.arguments.offline_model_path))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)
        utils.check_file_or_directory_path((os.path.realpath(self.arguments.cann_path)), True)
        atc_command_file_path = self.get_atc_path()
        utils.check_file_or_directory_path(atc_command_file_path)
        output_json_path = os.path.join(self.arguments.out_path, "model", model_name + ".json")
        # do the atc command to convert om to json
        utils.logger.info('Start to converting the model to json')
        atc_cmd = [atc_command_file_path, "--mode=1", "--om=" + self.arguments.offline_model_path,
                   "--json=" + output_json_path]
        utils.logger.info("ATC command line %s" % " ".join(atc_cmd))
        utils.execute_command(atc_cmd, True)
        utils.logger.info("Complete model conversion to json %s." % output_json_path)
        return output_json_path

    def get_atc_path(self):
        atc_command_file_path = os.path.join(self.arguments.cann_path, ATC_FILE_PATH)
        if os.path.exists(atc_command_file_path):
            return atc_command_file_path
        else:
            return os.path.join(self.arguments.cann_path, OLD_ATC_FILE_PATH)
