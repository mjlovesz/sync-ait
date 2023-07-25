#!/bin/bash

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

declare -i ret_ok=0
declare -i ret_invalid_args=1
CUR_PATH=$(dirname $(readlink -f "$0"))
. $CUR_PATH/utils.sh
set -x
set -e
MSAME_PATH=$CUR_PATH/msame
function get_msame_file()
{
    get_arch=`arch`
    if [[ $get_arch =~ "x86_64" ]];then
        echo "arch x86_64"
        local convert_url="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame"
        wget $convert_url -O $1 --no-check-certificate
    elif [[ $get_arch =~ "aarch64" ]];then
        echo "arch arm64"
        local convert_url="https://aisbench.obs.myhuaweicloud.com/packet/msame/arm/msame"
        wget $convert_url -O $1 --no-check-certificate
    else
        echo "unknown!!"l
    fi
}

main() {
    if [ $# -lt 2 ]; then
        echo "at least one parameter. for example: bash test.sh Ascend310P3 python3"
        return $ret_invalid_args
    fi

    export SOC_VERSION=${1:-"Ascend310P3"}
    export PYTHON_COMMAND=${2:-"python3"}
    export BENCKMARK_DT_MODE=${3:-"simple"}
    export PYTHONPATH=$CUR_PATH:$PYTHONPATH

    get_msame_file $MSAME_PATH || { echo "get msame bin file failed";return 1; }
    chmod 750 $MSAME_PATH
    # export MSAME_BIN_PATH=$CUR_PATH/../../../../../../tools/msame/out/msame
    export MSAME_BIN_PATH=$MSAME_PATH
    [ -f $MSAME_BIN_PATH ] || { echo "not find msame:$MSAME_BIN_PATH please check"; return $ret_invalid_args; }

    check_python_package_is_install $PYTHON_COMMAND "aclruntime" || {
        echo "aclruntime package install failed please install or source set_env.sh"
        return $ret_invalid_args
    }

    bash -x $CUR_PATH/get_pth_resnet50_data.sh $SOC_VERSION $PYTHON_COMMAND $BENCKMARK_DT_MODE
    #bash -x $CUR_PATH/get_pth_resnet101_data.sh $SOC_VERSION $PYTHON_COMMAND
    #bash -x $CUR_PATH/get_pth_inception_v3_data.sh $SOC_VERSION $PYTHON_COMMAND
    ${PYTHON_COMMAND} generate_pipeline_datasets.py

    if [ $BENCKMARK_DT_MODE == "full" ];then
        bash -x $CUR_PATH/get_bert_data.sh $SOC_VERSION $PYTHON_COMMAND
        bash -x $CUR_PATH/get_yolo_data.sh $SOC_VERSION $PYTHON_COMMAND
        bash -x $CUR_PATH/get_pth_crnn_data.sh $SOC_VERSION $PYTHON_COMMAND
    fi

    if [ $BENCKMARK_DT_MODE == "full" ];then
        echo "run DT in full mode"
        ${PYTHON_COMMAND} -m pytest -s $CUR_PATH/UT/
        ${PYTHON_COMMAND} -m pytest -s $CUR_PATH/ST/
    else
        echo "run DT in simple mode"
        ${PYTHON_COMMAND} -m pytest -x $CUR_PATH/UT_SIMPLE/
        ${PYTHON_COMMAND} -m pytest -x $CUR_PATH/ST_SIMPLE/
    fi

    return $ret_ok
}

main "$@"
exit $?
