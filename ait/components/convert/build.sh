#!/usr/bin/env bash
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

CONVERT_DIR=$(dirname $(readlink -f $0))
cd ${CONVERT_DIR}/aie_runtime/cpp
rm -rf build && mkdir build && cd build && cmake .. && make -j

AIT_CONVERT=${CONVERT_DIR}/aie_runtime/cpp/build/ait_convert

AIE_DIR=$(dirname $(python3 -c "import aie_runtime;print(aie_runtime.__file__)"))

if [ -f ${AIT_CONVERT} ];then
  cp ${AIT_CONVERT} ${AIE_DIR}
  else
    echo "Error: Build ait_convert failed."
fi
