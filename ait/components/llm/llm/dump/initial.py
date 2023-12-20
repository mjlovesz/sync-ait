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
import site

from llm.common.constant import ATB_SAVE_TENSOR_TIME, ATB_SAVE_TENSOR_IDS, \
    ATB_SAVE_TENSOR_RUNNER, ATB_SAVE_TENSOR, ATB_SAVE_TENSOR_RANGE, \
    ATB_SAVE_TILING, LD_PRELOAD, ATB_OUTPUT_DIR, ATB_SAVE_CHILD, ATB_SAVE_TENSOR_PART


def init_dump_task(args):
    if args.save_desc:
        os.environ[ATB_SAVE_TENSOR] = "2"
    else:
        os.environ[ATB_SAVE_TENSOR] = "1"
    
    os.environ[ATB_SAVE_TENSOR_TIME] = str(args.time)
    if args.ids:
        os.environ[ATB_SAVE_TENSOR_IDS] = str(args.ids)
    if args.opname:
        os.environ[ATB_SAVE_TENSOR_RUNNER] = str(args.opname).lower()
    if args.output:
        if args.output.endswith('/'):
            os.environ[ATB_OUTPUT_DIR] = str(args.output)
        else:
            os.environ[ATB_OUTPUT_DIR] = str(args.output) + '/'
        atb_dump_path = os.path.join(args.output, 'atb_temp', 'tensors')
        os.makedirs(atb_dump_path, exist_ok=True)
    os.environ[ATB_SAVE_CHILD] = "1" if args.child else "0"
    os.environ[ATB_SAVE_TENSOR_RANGE] = str(args.range)
    os.environ[ATB_SAVE_TILING] = "1" if args.tiling else "0"
    os.environ[ATB_SAVE_TENSOR_PART] = str(args.save_tensor_part)
    ld_preload = os.getenv(LD_PRELOAD)
    ld_preload = ld_preload or ""
    save_tensor_so_path = os.path.join(site.getsitepackages()[0], "llm/dump/backend/lib", \
                                       "libatb_probe.so")
    os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload


def clear_dump_task():
    pass