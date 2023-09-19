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
from msquickcmp.pta_acl_cmp.cmp_algorithm import cmp_alg_map

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."

PTA = "pta"
ACL = "acl"
DATA_ID = 'data_id'
PTA_DATA_PATH = 'pta_data_path'
ACL_DATA_PATH = 'acl_data_path'
PTA_DTYPE = "pta_dtype"
PTA_SHAPE = "pta_shape"
PTA_MAX_VALUE = "pta_max_value"
PTA_MIN_VALUE = "pta_min_value"
PTA_MEAN_VALUE = "pta_mean_value"
PTA_STACK = "pta_stack"
ACL_DTYPE = "acl_dtype"
ACL_SHAPE = "acl_shape"
ACL_MAX_VALUE = "acl_max_value"
ACL_MIN_VALUE = "acl_min_value"
ACL_MEAN_VALUE = "acl_mean_value"
ACL_STACK = "acl_stack"
GOLDEN_DATA_PATH = 'golden_data_path'
GOLDEN_DTYPE = 'golden_dtype'
GOLDEN_SHAPE = 'golden_shape'
GOLDEN_MAX_VALUE = "golden_max_value"
GOLDEN_MIN_VALUE = "golden_min_value"
GOLDEN_MEAN_VALUE = "golden_mean_value"
GOLDEN_STACK = "golden_stack"
CMP_FLAG = "cmp_flag"
CMP_FAIL_REASON = "cmp_fail_reason"
CSV_HEADER = [DATA_ID, PTA_DATA_PATH, PTA_DTYPE, PTA_SHAPE, PTA_MAX_VALUE, PTA_MIN_VALUE, PTA_MEAN_VALUE,
              ACL_DATA_PATH, ACL_DTYPE, ACL_SHAPE, ACL_MAX_VALUE, ACL_MIN_VALUE, ACL_MEAN_VALUE, CMP_FLAG]
CSV_GOLDEN_HEADER = [DATA_ID, GOLDEN_DATA_PATH, GOLDEN_DTYPE, GOLDEN_SHAPE, GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE, GOLDEN_MEAN_VALUE,
              ACL_DATA_PATH, ACL_DTYPE, ACL_SHAPE, ACL_MAX_VALUE, ACL_MIN_VALUE, ACL_MEAN_VALUE, CMP_FLAG]
CSV_HEADER.extend(list(cmp_alg_map.keys()))
CSV_HEADER.append(CMP_FAIL_REASON)
CSV_GOLDEN_HEADER.extend(list(cmp_alg_map.keys()))
CSV_GOLDEN_HEADER.append(CMP_FAIL_REASON)

MODEL_INFER_TASK_ID = "AIT_CMP_TASK_ID"
AIT_CMP_TASK_DIR = 'AIT_CMP_TASK_DIR'
AIT_CMP_TASK = "AIT_CMP_TASK"
AIT_CMP_TASK_PID = "AIT_CMP_TASK_PID"
AIT_IS_SAVE_MD5 = "AIT_IS_SAVE_MD5"
AIT_DIALOG_DUMP_PATH = "AIT_DIALOG_DUMP_PATH"
LD_PRELOAD = "LD_PRELOAD"

ACL_DATA_MAP_FILE = "ait_compare_acl_map.txt"

ACLTRANSFORMER_SAVE_TENSOR_MAX = "ACLTRANSFORMER_SAVE_TENSOR_MAX"
ACLTRANSFORMER_SAVE_TENSOR = "ACLTRANSFORMER_SAVE_TENSOR"
MAX_TOKEN_NUM = "10000"
AIT_DUMP_PATH = "AIT_DUMP_PATH"
TOKEN_ID = "token_id"
