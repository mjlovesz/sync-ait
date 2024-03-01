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

import os

from llm.common.constant import ATB_OUTPUT_DIR, ATB_CHECK_TYPE, CHECK_TYPE_MAPPING, ATB_EXIT
from llm.common.log import logger

    
def init_error_check(args) -> None:
    # type
    os.environ[ATB_CHECK_TYPE] = ''.join(CHECK_TYPE_MAPPING[type_] for type_ in args.type)
    
    # output_dir
    output_dir = args.output
    if not output_dir:
        default_dir = os.path.join(os.getcwd(), r'ait_err_check', r'overflow')
        logger.warning("Output directory is not provided."
                       "Results will be stored under the default directory instead."
                       f"Please refer to the directory {default_dir}")
        os.makedirs(default_dir, exist_ok=True)
        os.environ[ATB_OUTPUT_DIR] = default_dir
    else:
        output_dir = os.path.join(output_dir, r'ait_err_check', r'overflow')
        output_dir = os.path.abspath(output_dir)
        os.makedirs(default_dir, exist_ok=True)
        os.environ[ATB_OUTPUT_DIR] = output_dir

    # break
    os.environ[ATB_EXIT] = '1' if args.exit else '0'
    
    logger.info("Error check is ready. Inference processing.")
    