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
    local convert_url="https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet18_for_PyTorch/resnet18_pth2onnx.py"
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
    local _input_path=$1
    local _model_name=$2
    local _input_shape=$3

    local _onnx_path="$_input_path/${_model_name}.onnx"
    local _om_path_pre="$_input_path/${_model_name}_bs1"
    local _om_path="$_om_path_pre.om"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$_onnx_path --output=$_om_path_pre --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION"
        $_cmd || { echo "static model atc run failed";return $ret_failed; }
    fi
}

# 动态batch转换
convert_dymbatch_om()
{
    local _input_path=$1
    local _model_name=$2
    local _input_shape=$3
    local _dymbatch=$4

    local _onnx_path="$_input_path/${_model_name}.onnx"
    local _om_path_pre="$_input_path/${_model_name}_dymbatch"
    local _om_path="$_om_path_pre.om"

    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$_onnx_path --output=$_om_path_pre --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION --dynamic_batch_size=$_dymbatch"
        $_cmd || { echo "dymbatch model atc run failed";return $ret_failed; }
    fi
}

# 动态宽高 转换
convert_dymhw_om()
{
    local _input_path=$1
    local _model_name=$2
    local _input_shape=$3
    local _dymhw=$4

    local _onnx_path="$_input_path/${_model_name}.onnx"
    local _om_path_pre="$_input_path/${_model_name}_dymwh"
    local _om_path="$_om_path_pre.om"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$_onnx_path --output=$_om_path_pre --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION --dynamic_image_size=$_dymhw"
        $_cmd || { echo "dymwh model atc run failed";return $ret_failed; }
    fi
}

# 动态dims转换
convert_dymdims_om()
{
    local _input_path=$1
    local _model_name=$2
    local _input_shape=$3
    local _dymdims=$4

    local _onnx_path="$_input_path/${_model_name}.onnx"
    local _om_path_pre="$_input_path/${_model_name}_dymdims"
    local _om_path="$_om_path_pre.om"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$_onnx_path --output=$_om_path_pre --input_format=ND --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION --dynamic_dims=$_dymdims"
        $_cmd || { echo "dymwh model atc run failed";return $ret_failed; }
    fi
}

# 动态shape转换
convert_dymshape_om()
{
    local _input_path=$1
    local _model_name=$2
    local _input_shape=$3

    local _onnx_path="$_input_path/${_model_name}.onnx"
    local _om_path_pre="$_input_path/${_model_name}_dymshape"
    local _om_path="$_om_path_pre.om"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$_onnx_path --output=$_om_path_pre --framework=5 \
            --input_shape_range=$_input_shape --soc_version=$SOC_VERSION"
        $_cmd || { echo "dymwh model atc run failed";return $ret_failed; }
    fi
}

main()
{
    get_npu_type || { echo "get npu type failed";return $ret_failed; }
    PYTHON_COMMAND="python3"
    SAMPLE_RES_PATH=$CUR_PATH/sampledata/resnet18/model
    SAMPLE_ADD_PATH=$CUR_PATH/sampledata/add_model/model
    ADD_ONNX_PATH=$SAMPLE_ADD_PATH/add_model.onnx

    [ -d $SAMPLE_RES_PATH ] || mkdir -p $SAMPLE_RES_PATH
    [ -d $SAMPLE_ADD_PATH ] || mkdir -p $SAMPLE_ADD_PATH

    if [[ ! -f $ADD_ONNX_PATH ]]; then
        python3 generate_add_model.py || { echo "generate add onnx failed";return $ret_failed; }
        mv $CUR_PATH/add_model.onnx $SAMPLE_ADD_PATH/ || { echo "move add onnx failed";return $ret_failed; }
    fi

    model_url="https://download.pytorch.org/models/resnet18-f37072fd.pth"
    resnet_pth_file="$SAMPLE_RES_PATH/pth_resnet18.pth"
    if [ ! -f $resnet_pth_file ]; then
        try_download_url $model_url $resnet_pth_file || { echo "donwload stubs failed";return $ret_failed; }
    fi
    resnet_onnx_file="$SAMPLE_RES_PATH/pth_resnet18.onnx"
    if [ ! -f $resnet_onnx_file ]; then
        convert_file_path=$SAMPLE_RES_PATH/resnet18_pth2onnx.py
        get_convert_file $convert_file_palsth || { echo "get convert file failed";return $ret_failed; }
        chmod 750 $convert_file_path
        cd $SAMPLE_RES_PATH
        $PYTHON_COMMAND $convert_file_path --checkpoint $resnet_pth_file --save_dir $SAMPLE_RES_PATH/resnet18.onnx || { echo "convert pth to onnx failed";return $ret_failed; }
        mv $SAMPLE_RES_PATH/resnet18.onnx $resnet_onnx_file
        cd -
    fi

    echo "Start convert pth_resnet18.onnx to om, it may take a few minutes"
    model_kind="pth_resnet18"

    input_shape="image:1,3,224,224"
    convert_staticbatch_om $SAMPLE_RES_PATH $model_kind $input_shape || { echo "convert static $model_kind om failed";return $ret_failed; }
    input_shape="image:-1,3,224,224"
    dymbatch="1,2,4,8"
    convert_dymbatch_om $SAMPLE_RES_PATH $model_kind $input_shape $dymbatch || { echo "convert dymbatch $model_kind om failed";return $ret_failed; }
    input_shape="image:1,3,-1,-1"
    dymhw="224,224;448,448"
    convert_dymhw_om $SAMPLE_RES_PATH $model_kind $input_shape $dymhw || { echo "convert dymhw $model_kind om failed";return $ret_failed; }
    input_shape="image:-1,3,-1,-1"
    dymdims="1,224,224;8,448,448"
    convert_dymdim_om $SAMPLE_RES_PATH $model_kind $input_shape $dymdims || { echo "convert dymdim $model_kind om failed";return $ret_failed; }

    echo "Start convert add_model.onnx to om, it may take a few minutes"
    model_kind="add_model"

    input_shape="input1:1,3,32,32;input2:1,3,32,32"
    convert_staticbatch_om $SAMPLE_ADD_PATH $model_kind $input_shape || { echo "convert static $model_kind om failed";return $ret_failed; }
    input_shape="input1:-1,3,32,32;input2:-1,3,32,32"
    dymbatch="1,2,4,8"
    convert_dymbatch_om $SAMPLE_ADD_PATH $model_kind $input_shape $dymbatch || { echo "convert dymbatch $model_kind om failed";return $ret_failed; }
    input_shape="input1:1,3,-1,-1;input2:1,3,-1,-1"
    dymhw="32,32;64,64"
    convert_dymhw_om $SAMPLE_ADD_PATH $model_kind $input_shape $dymhw || { echo "convert dymhw $model_kind om failed";return $ret_failed; }
    input_shape="input1:-1,3,-1,-1;input2:-1,3,-1,-1"
    dymdims="1,32,32,1,32,32;4,64,64,4,64,64"
    convert_dymdim_om $SAMPLE_ADD_PATH $model_kind $input_shape $dymdims || { echo "convert dymdim $model_kind om failed";return $ret_failed; }



    # dymshapes 310 不支持，310P支持
    if [ $SOC_VERSION != "Ascend310" ]; then
        echo "dymshape enabled"
        dymshapes="[1~16,3,200~300,200~300]"
        convert_dymshape_om $SAMPLE_RES_PATH $model_kind $dymshapes || { echo "convert dymshape resnet18 om failed";return $ret_failed; }
        if [ ! -f $SAMPLE_RES_PATH/pth_resnet18_dymshape.om ]; then
            mv $SAMPLE_RES_PATH/pth_resnet18_dymshape*.om $SAMPLE_RES_PATH/pth_resnet18_dymshape.om
        fi
        dymshapes="input1:[1~4,3,32~64,32~64];input2:[1~4,3,32~64,32~64]"
        convert_dymshape_om $SAMPLE_ADD_PATH $model_kind $dymshapes || { echo "convert dymshape add_model om failed";return $ret_failed; }
        if [ ! -f $SAMPLE_ADD_PATH/add_model_dymshape.om ]; then
            mv $SAMPLE_ADD_PATH/add_model_dymshape*.om $SAMPLE_ADD_PATH/add_model_dymshape.om
        fi
    fi
    echo "All atc finished!"

    $PYTHON_COMMAND generate_datasets.py || { echo "generate datasets failed";return $ret_failed; }
}

main "$@"
exit $?
