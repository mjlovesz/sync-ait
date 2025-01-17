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

#!/bin/bash
CUR_PATH=$(dirname $(readlink -f "$0"))

try_download_url(){
    local _url=$1
    local _packet=$2
    cmd="wget $_url --no-check-certificate -O $_packet"
    $cmd #>/dev/null 2>&1
    ret=$?
    if [ "$ret" == 0 -a -s "$_packet" ];then
        echo "download cmd:$cmd targetfile:$ OK"
    else
        echo "downlaod targetfile by $cmd Failed please check network or manual download to target file"
        return 1
    fi
}

convert_staticbatch_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _staticbatch=$3
    local _framework=3

    # 静态batch转换
    for batchsize in $_staticbatch; do
        local _input_shape="$ids_name:$batchsize,384;$mask_name:$batchsize,384;$seg_name:$batchsize,384"
        local _pre_name=${_input_file%.*}
        local _om_path_pre="${_pre_name}_bs${batchsize}"
        local _om_path="$_om_path_pre.om"
        if [ ! -f $_om_path ];then
            local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework --input_shape=$_input_shape --soc_version=$_soc_version"
            $_cmd || { echo "atc run $_cmd failed"; return 1; }
        fi
    done
}

# 动态batch转换
convert_dymbatch_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _dymbatch=$3
    local _framework=3

    local _batchsize="-1"
    local _input_shape="$ids_name:$_batchsize,384;$mask_name:$_batchsize,384;$seg_name:$_batchsize,384"
    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymbatch"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ];then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework --input_shape=$_input_shape -soc_version=$_soc_version --dynamic_batch_size=$_dymbatch"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

main()
{
    SOC_VERSION=${1:-"Ascend310P3"}
    PYTHON_COMMAND=${2:-"python3"}
    TESTDATA_PATH=$CUR_PATH/testdata/bert/model
    [ -d $TESTDATA_PATH ] || { mkdir -p $TESTDATA_PATH;chmod 750 $TESTDATA_PATH; }
    [ -d $TESTDATA_PATH/tmp ] || { mkdir -p $TESTDATA_PATH/tmp/;chmod 750 $TESTDATA_PATH; }

    model_url="https://ascend-repo-modelzoo.obs.myhuaweicloud.com/model/ATC%20BERT_BASE_SQuAD1.1%28FP16%29%20from%20Tensorflow-Ascend310/zh/1.1/ATC%20BERT_BASE_SQuAD1.1%28FP16%29%20from%20Tensorflow-Ascend310.zip"
    bert_pb_file="$TESTDATA_PATH/pth_bert.pb"
    if [ ! -f $bert_pb_file ]; then
        try_download_url $model_url $TESTDATA_PATH/tmp/a.zip || { echo "donwload stubs failed";return 1; }
        unzip $TESTDATA_PATH/tmp/a.zip -d $TESTDATA_PATH/tmp/
        origin_pb_file="BERT_Base_SQuAD1_1_BatchSize_None.pb"
        find $TESTDATA_PATH/tmp/ -name $origin_pb_file | xargs -I {} cp {} $bert_pb_file
    fi

    [ -f $bert_pb_file ] || { echo "find no $bert_pb_file return";return 1; }
    ids_name="input_ids"
    mask_name="input_mask"
    seg_name="segment_ids"

    staticbatch="1 2 4 8 16"
    convert_staticbatch_om $bert_pb_file $input_tensor_name $SOC_VERSION "${staticbatch[*]}" || { echo "convert static om failed";return 1; }
    dymbatch="1,2,4,8,16"
    convert_dymbatch_om $bert_pb_file $input_tensor_name $SOC_VERSION $dymbatch || { echo "convert dymbatch om failed";return 1; }
}

main "$@"
exit $?