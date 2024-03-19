#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

set -e
SCRIPT_DIR=$(cd $(dirname $0); pwd)

export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
export AIT_LLM_INSTALL_PATH="$(python3 -c 'import llm, os; print(os.path.dirname(os.path.abspath(llm.__file__)))')"

function fn_build_nlohmann_json()
{
    if [ -d "$SCRIPT_DIR/dependency/nlohmann" ]; then
        return
    fi
    if [ -d "$SCRIPT_DIR/cache" ]; then
        rm -rf $SCRIPT_DIR/cache
    fi
    mkdir -p $SCRIPT_DIR/cache
    cd $SCRIPT_DIR/cache
    wget --no-check-certificate https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    tar -zxvf ./v3.11.3.tar.gz
    cp -r ./json-3.11.3/include/nlohmann $SCRIPT_DIR/dependency
    rm -rf $SCRIPT_DIR/cache/
}

function fn_build_lib_opchecker()
{
    if [ -d "$SCRIPT_DIR/build" ]; then
        rm -rf $SCRIPT_DIR/build
    fi
    mkdir -p $SCRIPT_DIR/build
    cd $SCRIPT_DIR/build
    cmake .. -DCMAKE_INSTALL_PREFIX=$AIT_LLM_INSTALL_PATH/opcheck
    make -j4
    make install
}

function fn_main()
{
    fn_build_nlohmann_json
    fn_build_lib_opchecker
}
fn_main "$@"
