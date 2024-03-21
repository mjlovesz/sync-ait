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

from components.utils.file_open_check import FileStat
from ait_llm.common.constant import ATB_CUR_PID, LD_PRELOAD, ATB_PROB_LIB_WITH_ABI, ATB_PROB_LIB_WITHOUT_ABI, ATB_HOME_PATH, ASCEND_TOOLKIT_HOME, ATB_OUTPUT_DIR, ATB_CHECK_TYPE, CHECK_TYPE_MAPPING, ATB_EXIT
from ait_llm.common.log import logger
from ait_llm.dump.initial import is_use_cxx11
        
    
def init_error_check(args) -> None:
    # locate cann directory
    cann_path = os.environ.get(ASCEND_TOOLKIT_HOME, "/usr/local/Ascend/ascend-toolkit/latest")
    if not cann_path or not os.path.exists(cann_path):
        raise OSError("cann_path is invalid, please install cann-toolkit and set the environment variables.")

    cur_is_use_cxx11 = is_use_cxx11()
    logger.info("Info detected from ATB so is_use_cxx11: %s", cur_is_use_cxx11)
    
    save_tensor_so_name = ATB_PROB_LIB_WITH_ABI if cur_is_use_cxx11 else ATB_PROB_LIB_WITHOUT_ABI
    save_tensor_so_path = os.path.join(cann_path, "tools", "ait_backend", "dump", save_tensor_so_name)
    if not os.path.exists(save_tensor_so_path):
        raise OSError(f"{save_tensor_so_name} is not found in {cann_path}. Try installing the latest cann-toolkit")
    if not FileStat(save_tensor_so_path).is_basically_legal('read', strict_permission=True):
        raise OSError(f"{save_tensor_so_name} is illegal, group or others writable file stat is not permitted")

    logger.info("Append save_tensor_so_path: %s to LD_PRELOAD", save_tensor_so_path)
    ld_preload = os.getenv(LD_PRELOAD)
    ld_preload = ld_preload or ""
    os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload
    
    # type
    os.environ[ATB_CHECK_TYPE] = ''.join(CHECK_TYPE_MAPPING[type_] for type_ in args.type)
    
    # output_dir
    output_dir = args.output
    if not output_dir:
        # set default directory to current work directory
        output_dir = os.getcwd()
        logger.warning("Output directory is not provided. "
                       "Results will be stored under the default directory instead.")
    else:
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            logger.warning("Specified directory does not exist. directory creating...")
            os.makedirs(output_dir, mode=750, exist_ok=True)       
    os.environ[ATB_OUTPUT_DIR] = output_dir

    # exit
    os.environ[ATB_EXIT] = '1' if args.exit else '0'
    
    logger.info("Initialization finished. Inference processing.")
    