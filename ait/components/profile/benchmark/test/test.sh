#!/bin/bash

# Copyright 2022 Huawei Technologies Co., Ltd. All rights reserved.
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

declare -i ret_ok=0
declare -i ret_invalid_args=1
CUR_PATH=$(dirname $(readlink -f "$0"))
. $CUR_PATH/utils.sh
set -x
set -e

main() {
    if [ $# -lt 2 ]; then
        echo "at least one parameter. for example: bash test.sh Ascend310P3 python3"
        return $ret_invalid_args
    fi

    export SOC_VERSION=${1:-"Ascend310P3"}
    export PYTHON_COMMAND=${2:-"python3"}

    export MSAME_BIN_PATH=$CUR_PATH/../../../../msame/out/msame
    [ -f $MSAME_BIN_PATH ] || { echo "not find msame:$MSAME_BIN_PATH please check"; return $ret_invalid_args; }

    check_python_package_is_install $PYTHON_COMMAND "aclruntime" || {
        echo "aclruntime package install failed please install or source set_env.sh"
        return $ret_invalid_args
    }

    bash -x $CUR_PATH/get_pth_resnet50_data.sh $SOC_VERSION $PYTHON_COMMAND
    #bash -x $CUR_PATH/get_pth_resnet101_data.sh $SOC_VERSION $PYTHON_COMMAND
    #bash -x $CUR_PATH/get_pth_inception_v3_data.sh $SOC_VERSION $PYTHON_COMMAND
    bash -x $CUR_PATH/get_bert_data.sh $SOC_VERSION $PYTHON_COMMAND
    bash -x $CUR_PATH/get_yolo_data.sh $SOC_VERSION $PYTHON_COMMAND
    bash -x $CUR_PATH/get_pth_crnn_data.sh $SOC_VERSION $PYTHON_COMMAND
    ${PYTHON_COMMAND} -m pytest -s $CUR_PATH/ST/
    ${PYTHON_COMMAND} -m pytest -s $CUR_PATH/UT/

    return $ret_ok
}

main "$@"
exit $?
