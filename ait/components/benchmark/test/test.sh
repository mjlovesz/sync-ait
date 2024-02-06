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
declare -i ret_failed=1
declare -i ret_invalid_args=1
CUR_PATH=$(dirname $(readlink -f "$0"))

PYTHON_COMMAND="python3"
SOC_VERSION=`$PYTHON_COMMAND -c 'import acl; print(acl.get_soc_name())'`
BENCKMARK_DT_MODE="simple"

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

function chmod_file_data()
{
    chmod 750 $CUR_PATH/json_for_arg_test.json
    chmod -R 750 $CUR_PATH/aipp_config_files
}

main() {
    chmod_file_data
    echo "Usage: bash test.sh {SOC_VERSION} {PYTHON_COMMAND} {BENCKMARK_DT_MODE}"

    if [ $# -gt 1 ]; then
        SOC_VERSION=$1
    else if [ $# -gt 2 ]; then
        PYTHON_COMMAND=$2
    else if [ $# -gt 3 ]; then
        BENCKMARK_DT_MODE=$3
    fi

    echo "SOC_VERSION=$SOC_VERSION, PYTHON_COMMAND=$PYTHON_COMMAND, BENCKMARK_DT_MODE=$BENCKMARK_DT_MODE"
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
    if [ ! -n $AIT_BENCHMARK_DT_DATA_PATH ]; then
        echo "using $AIT_BENCHMARK_DT_DATA_PATH as dt data path"
        bash -x $CUR_PATH/get_pth_resnet50_data.sh $SOC_VERSION $PYTHON_COMMAND $BENCKMARK_DT_MODE
        bash -x $CUR_PATH/get_add_model_data.sh
    fi
    #bash -x $CUR_PATH/get_pth_resnet101_data.sh $SOC_VERSION $PYTHON_COMMAND
    #bash -x $CUR_PATH/get_pth_inception_v3_data.sh $SOC_VERSION $PYTHON_COMMAND
    ${PYTHON_COMMAND} $CUR_PATH/generate_pipeline_datasets.py

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
        ${PYTHON_COMMAND} -m pytest -x $CUR_PATH/UT_SIMPLE/ || { return $ret_failed; }
        ${PYTHON_COMMAND} -m pytest -x $CUR_PATH/ST_SIMPLE/ || { return $ret_failed; }
    fi

    return $ret_ok
}

main "$@"
exit $?
