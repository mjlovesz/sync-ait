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

if [ ! ${AIE_DIR} ];then
  echo "Error: Ascend Inference Engine is not installed."
  exit 1
fi

CONVERT_DIR=$(dirname $(readlink -f $0))
cd ${CONVERT_DIR}/model_convert/cpp
rm -rf build && mkdir build && cd build && cmake .. && make -j

AIE_CONVERT=${CONVERT_DIR}/model_convert/cpp/build/aie_convert

if [ ! "$(command -v pip3)" ];then
  echo "Error: pip3 is not installed."
  exit 1
fi

PIP3=$(readlink -f $(which pip3))
PIP3_DIR=$(dirname ${PIP3})
PYTHON=${PIP3_DIR}/python

AIE_RUNTIME_PATH=$(dirname $(${PYTHON} -c "import aie_runtime;print(aie_runtime.__file__)"))

if [ -f ${AIE_CONVERT} ];then
  cp ${AIT_CONVERT} ${AIE_RUNTIME_PATH}
  else
    echo "Error: Build aie_convert failed."
fi
