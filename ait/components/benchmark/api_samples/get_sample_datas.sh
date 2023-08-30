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
CUR_PATH=$(dirname $(readlink -f "$0"))
SOC_VERSION=""

try_download_url() {
    local _url=$1
    local _packet=$2
    cmd="wget $_url --no-check-certificate -O $_packet"
    $cmd #>/dev/null 2>&1
    ret=$?
    if [ "$ret" == 0 -a -s "$_packet" ]; then
        echo "download cmd:$cmd targetfile:$ OK"
    else
        echo "downlaod targetfile by $cmd Failed please check network or manual download to target file"
        return $ret_failed
    fi
}

function get_convert_file()
{
    local convert_url="https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/pth2onnx.py"
    wget $convert_url -O $1 --no-check-certificate
}

function get_npu_type()
{
    get_npu_310=`lspci | grep d100`
    get_npu_310P3=`lspci | grep d500`
    get_npu_310B=`lspci | grep d107`
    if [[ $get_npu_310 != "" ]];then
        SOC_VERSION="Ascend310"
        echo "npu is Ascend310, dymshape sample not supported"
    elif [[ $get_npu_310P3 != "" ]];then
        SOC_VERSION="Ascend310P3"
        echo "npu is Ascend310P3"
    elif [[ $get_npu_310B != "" ]];then
        SOC_VERSION="Ascend310B"
        echo "npu is Ascend310B"
    else
        return $ret_failed
    fi
}

convert_staticbatch_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _staticbatch=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    # 静态batch转换
    for batchsize in $_staticbatch; do
        local _input_shape="$_input_tensor_name:$batchsize,3,224,224"
        local _pre_name=${_input_file%.*}
        local _om_path_pre="${_pre_name}_bs${batchsize}"
        local _om_path="$_om_path_pre.om"
        if [ ! -f $_om_path ]; then
            local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
                --input_shape=$_input_shape --soc_version=$_soc_version \
                --input_format=NCHW --enable_small_channel=1"
            [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
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
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    local _input_shape="$_input_tensor_name:-1,3,224,224"
    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymbatch"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
        --input_shape=$_input_shape -soc_version=$_soc_version --dynamic_batch_size=$_dymbatch \
        --input_format=NCHW --enable_small_channel=1"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

# 动态宽高 转换
convert_dymhw_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _dymhw=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    local _input_shape="$_input_tensor_name:1,3,-1,-1"
    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymwh"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
        --input_shape=$_input_shape -soc_version=$_soc_version --dynamic_image_size=$_dymhw
        --input_format=NCHW --enable_small_channel=1"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

# 动态dims转换
convert_dymdim_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _dymdim=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    local _input_shape="$_input_tensor_name:-1,3,-1,-1"
    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymdim"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
            --input_shape=$_input_shape -soc_version=$_soc_version --input_format=ND --dynamic_dims=$_dymdim \
            --enable_small_channel=1"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

# 动态shape转换
convert_dymshape_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _dymshapes=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymshape"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
            --input_shape_range=$_input_tensor_name:$_dymshapes --soc_version=$_soc_version \
            --input_format=NCHW"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

main()
{
    get_npu_type() || { echo "get npu type failed";return $ret_failed; }
    PYTHON_COMMAND="python3"
    SAMPLEDATA_PATH=$CUR_PATH/sampledata/resnet18/model
    [ -d $SAMPLEDATA_PATH ] || mkdir -p $SAMPLEDATA_PATH

    model_url="https://download.pytorch.org/models/resnet18-f37072fd.pth"
    resnet_pth_file="$SAMPLEDATA_PATH/pth_resnet18.pth"
    if [ ! -f $resnet_pth_file ]; then
        try_download_url $model_url $resnet_pth_file || { echo "donwload stubs failed";return 1; }
    fi
    resnet_onnx_file="$SAMPLEDATA_PATH/pth_resnet18.onnx"
    input_tensor_name="image"
    if [ ! -f $resnet_onnx_file ]; then
        convert_file_path=$SAMPLEDATA_PATH/resnet18_pth2onnx.py
        get_convert_file $convert_file_path || { echo "get convert file failed";return $ret_failed; }
        chmod 750 $convert_file_path
        cd $SAMPLEDATA_PATH
        python3 $convert_file_path --checkpoint $pth_file --save_dir $CUR_PATH/onnx/resnet18.onnx || { echo "convert pth to onnx failed";return $ret_failed; }
        mv $SAMPLEDATA_PATH/resnet18.onnx $resnet_onnx_file
        cd -
    fi

    staticbatch="4"
    convert_staticbatch_om $resnet_onnx_file $SOC_VERSION "${staticbatch[*]}" $input_tensor_name || { echo "convert static om failed";return 1; }

    dymbatch="1,2,4,8,16"
    convert_dymbatch_om $resnet_onnx_file $SOC_VERSION $dymbatch $input_tensor_name || { echo "convert dymbatch om failed";return 1; }

    dymhw="224,224;448,448"
    convert_dymhw_om $resnet_onnx_file $SOC_VERSION $dymhw $input_tensor_name || { echo "convert dymhw om failed";return 1; }

    dymdims="1,224,224;8,448,448"
    convert_dymdim_om $resnet_onnx_file $SOC_VERSION $dymdims $input_tensor_name || { echo "convert dymdim om failed";return 1; }


    # dymshapes 310 不支持，310P支持
    if [ $SOC_VERSION != "Ascend310" ]; then
        echo "test dymshape enabled"
        dymshapes="[1~16,3,200~300,200~300]"
        convert_dymshape_om $resnet_onnx_file $SOC_VERSION $dymshapes $input_tensor_name || { echo "convert dymshape om failed";return 1; }
        if [ ! -f $SAMPLEDATA_PATH/pth_resnet18_dymshape.om ]; then
            mv $SAMPLEDATA_PATH/pth_resnet18_dymshape*.om $SAMPLEDATA_PATH/pth_resnet18_dymshape.om
        fi
    fi
}

main "$@"
exit $?
