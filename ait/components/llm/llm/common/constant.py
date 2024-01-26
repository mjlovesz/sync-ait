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
from llm.compare.cmp_algorithm import CMP_ALG_MAP

ATB_HOME_PATH = "ATB_HOME_PATH"
ATB_SAVE_TENSOR_TIME = "ATB_SAVE_TENSOR_TIME"
ATB_SAVE_TENSOR_IDS = "ATB_SAVE_TENSOR_IDS"
ATB_SAVE_TENSOR_RUNNER = "ATB_SAVE_TENSOR_RUNNER"
ATB_SAVE_TENSOR = "ATB_SAVE_TENSOR"
ATB_SAVE_TENSOR_RANGE = "ATB_SAVE_TENSOR_RANGE"
ATB_SAVE_TILING = "ATB_SAVE_TILING"
ATB_OUTPUT_DIR = "ATB_OUTPUT_DIR"
ATB_SAVE_CHILD = "ATB_SAVE_CHILD"
ATB_SAVE_TENSOR_PART = "ATB_SAVE_TENSOR_PART"
ATB_SAVE_CPU_PROFILING = "ATB_SAVE_CPU_PROFILING"
ATB_SAVE_OPERATION_INFO = "ATB_SAVE_OPERATION_INFO"
ATB_SAVE_KERNEL_INFO = "ATB_SAVE_KERNEL_INFO"
LD_PRELOAD = "LD_PRELOAD"
LOG_TO_STDOUT = "LOG_TO_STDOUT"

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."

MAX_DATA_SIZE = 2 * 1024 * 1024 * 1024  # 2G

ASCEND_TOOLKIT_HOME = "ASCEND_TOOLKIT_HOME"
ATB_PROB_LIB_WITH_ABI = "libatb_probe_abi1.so"
ATB_PROB_LIB_WITHOUT_ABI = "libatb_probe_abi0.so"

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."

PTA = "pta"
MY = "my"
DATA_ID = 'data_id'
TOKEN_ID = "token_id"
MY_DATA_PATH = 'my_data_path'
MY_DTYPE = "my_dtype"
MY_SHAPE = "my_shape"
MY_MAX_VALUE = "my_max_value"
MY_MIN_VALUE = "my_min_value"
MY_MEAN_VALUE = "my_mean_value"
MY_STACK = "my_stack"
GOLDEN_DATA_PATH = 'golden_data_path'
GOLDEN_DTYPE = 'golden_dtype'
GOLDEN_SHAPE = 'golden_shape'
GOLDEN_MAX_VALUE = "golden_max_value"
GOLDEN_MIN_VALUE = "golden_min_value"
GOLDEN_MEAN_VALUE = "golden_mean_value"
GOLDEN_STACK = "golden_stack"
CMP_FAIL_REASON = "cmp_fail_reason"
CSV_GOLDEN_HEADER = [TOKEN_ID, DATA_ID, GOLDEN_DATA_PATH, GOLDEN_DTYPE, GOLDEN_SHAPE, GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE, GOLDEN_MEAN_VALUE,
              MY_DATA_PATH, MY_DTYPE, MY_SHAPE, MY_MAX_VALUE, MY_MIN_VALUE, MY_MEAN_VALUE]
CSV_GOLDEN_HEADER.extend(list(CMP_ALG_MAP.keys()))
CSV_GOLDEN_HEADER.append(CMP_FAIL_REASON)

MODEL_INFER_TASK_ID = "AIT_CMP_TASK_ID"
AIT_CMP_TASK_DIR = 'AIT_CMP_TASK_DIR'
AIT_CMP_TASK = "AIT_CMP_TASK"
AIT_CMP_TASK_PID = "AIT_CMP_TASK_PID"
AIT_IS_SAVE_MD5 = "AIT_IS_SAVE_MD5"
AIT_DIALOG_DUMP_PATH = "AIT_DIALOG_DUMP_PATH"
AIT_DUMP_CLEAN = "AIT_DUMP_CLEAN"
LD_PRELOAD = "LD_PRELOAD"

MY_DATA_MAP_FILE = "ait_compare_my_map.txt"

MYTRANSFORMER_SAVE_TENSOR_MAX = "MYTRANSFORMER_SAVE_TENSOR_MAX"
MYTRANSFORMER_SAVE_TENSOR = "MYTRANSFORMER_SAVE_TENSOR"
MAX_TOKEN_NUM = "10000"
AIT_DUMP_PATH = "AIT_DUMP_PATH"