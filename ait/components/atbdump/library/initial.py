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
import shutil
import site

from constant import ATB_SAVE_TENSOR_TIME, ATB_SAVE_TENSOR_IDS, ATB_SAVE_TENSOR_RUNNER, ATB_SAVE_TENSOR, \
    ATB_SAVE_TENSOR_RANGE, ATB_SAVE_TILING, LD_PRELOAD

def init_dump_task(args):
    if args.save_desc:
        os.environ[ATB_SAVE_TENSOR] = "2"
    else:
        os.environ[ATB_SAVE_TENSOR] = "1"
    
    os.environ[ATB_SAVE_TENSOR_TIME] = args.time
    os.environ[ATB_SAVE_TENSOR_IDS] = args.ids
    os.environ[ATB_SAVE_TENSOR_RUNNER] = args.opname
    
    os.environ[ATB_SAVE_TENSOR_RANGE] = args.range
    os.environ[ATB_SAVE_TILING] = args.tiling
    ld_preload = os.getenv(LD_PRELOAD)
    ld_preload = ld_preload or ""
    save_tensor_so_path = os.path.join(site.getsitepackages()[0], "msquickcmp", "libatb_probe.so")
    os.environ[LD_PRELOAD] = save_tensor_so_path + ":" + ld_preload


def clear_aclcmp_task():
    pass