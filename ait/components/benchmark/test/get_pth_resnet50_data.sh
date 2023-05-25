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
        return 1
    fi
}

function get_convert_file()
{
    local convert_url="https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/pth2onnx.py"
    wget $convert_url -O $1 --no-check-certificate
}

function get_aippConfig_file()
{
    local aipp_config_url="https://gitee.com/ascend/ModelZoo-PyTorch/raw/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer/aipp_resnet50.aippconfig"
    wget $aipp_config_url -O $1 --no-check-certificate
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

convert_dymaipp_staticbatch_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _staticbatch=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    # 带动态aipp配置的静态batch转换
    for batchsize in $_staticbatch; do
        local _input_shape="$_input_tensor_name:$batchsize,3,224,224"
        local _pre_name=${_input_file%.*}
        local _om_path_pre="${_pre_name}_bs${batchsize}_dymaipp_stcbatch"
        local _om_path="$_om_path_pre.om"
        if [ ! -f $_om_path ]; then
            local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
                --input_shape=$_input_shape --soc_version=$_soc_version \
                --input_format=NCHW"
            [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
            $_cmd || { echo "atc run $_cmd failed"; return 1; }
        fi
    done
}

convert_dymaipp_dymbatch_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _dymbatch=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    local _input_shape="$_input_tensor_name:-1,3,224,224"
    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymaipp_dymbatch"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
        --input_shape=$_input_shape --soc_version=$_soc_version --dynamic_batch_size=$_dymbatch \
        --input_format=NCHW"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

convert_dymaipp_dymhw_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _dymhw=$3
    local _input_tensor_name=$4
    local _aippconfig=$5
    local _framework=5

    local _input_shape="$_input_tensor_name:1,3,-1,-1"
    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}_dymaipp_dymwh"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
        --input_shape=$_input_shape --soc_version=$_soc_version --dynamic_image_size=$_dymhw
        --input_format=NCHW"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

convert_multi_dymaipp_om()
{
    local _input_file=$1
    local _soc_version=$2
    local _aippconfig=$3
    local _framework=5

    local _pre_name=${_input_file%.*}
    local _om_path_pre="${_pre_name}"
    local _om_path="$_om_path_pre.om"

    if [ ! -f $_om_path ]; then
        local _cmd="atc --model=$_input_file --output=$_om_path_pre --framework=$_framework \
        --soc_version=$_soc_version"
        [ "$_aippconfig" != "" ] && _cmd="$_cmd --insert_op_conf=$_aippconfig"
        $_cmd || { echo "atc run $_cmd failed"; return 1; }
    fi
}

main()
{
    SOC_VERSION=${1:-"Ascend310P3"}
    PYTHON_COMMAND=${2:-"python3"}
    TESTDATA_PATH=$CUR_PATH/testdata/resnet50/model
    [ -d $TESTDATA_PATH ] || mkdir -p $TESTDATA_PATH

    model_url="https://download.pytorch.org/models/resnet50-0676ba61.pth"
    resnet_pth_file="$TESTDATA_PATH/pth_resnet50.pth"
    if [ ! -f $resnet_pth_file ]; then
        try_download_url $model_url $resnet_pth_file || { echo "donwload stubs failed";return 1; }
    fi

    cp $CUR_PATH/create_multi_inputs_onnx.py $TESTDATA_PATH/create_multi_inputs_onnx.py
    resnet_onnx_file="$TESTDATA_PATH/pth_resnet50.onnx"
    two_dymaipp_onnx_file="$TESTDATA_PATH/multi_dym_aipp_model.onnx"
    input_tensor_name="actual_input_1"
    if [ ! -f $resnet_onnx_file ]; then
        # generate convert_pth_to_onnx.py
        CONVERT_FILE_PATH=$TESTDATA_PATH/resnet50_convert_pth_to_onnx.py
        get_convert_file $CONVERT_FILE_PATH || { echo "get convert file failed";return 1; }
        cd $TESTDATA_PATH/
        $PYTHON_COMMAND $CONVERT_FILE_PATH $resnet_pth_file || { echo "convert pth to onnx failed";return 1; }
        mv $TESTDATA_PATH/resnet50_official.onnx $resnet_onnx_file
        cd -
    fi
    if [ ! -f $two_dymaipp_onnx_file ]; then
        cd $TESTDATA_PATH/
        $PYTHON_COMMAND $TESTDATA_PATH/create_multi_inputs_onnx.py || { echo "create onnx which supports two dymaipp_input failed";return 1; }
        mv $TESTDATA_PATH/multi_dym_aipp_model.onnx $two_dymaipp_onnx_file
        cd -
    fi

    AIPPCONFIG_FILE_PATH=$TESTDATA_PATH/aipp_resnet50.aippconfig
    DYM_AIPPCONFIG_FILE_PATH=$CUR_PATH/dym_aipp_cfg.config
    MULTI_DYM_AIPPCONFIG_FILE_PATH=$CUR_PATH/multi_dym_aipp_cfg.config
    get_aippConfig_file $AIPPCONFIG_FILE_PATH || { echo "get aipp file failed";return 1; }

    staticbatch="1 2 4 8 16"
    convert_staticbatch_om $resnet_onnx_file $SOC_VERSION "${staticbatch[*]}" $input_tensor_name $AIPPCONFIG_FILE_PATH || { echo "convert static om failed";return 1; }
    staticbatch="1 2 4"
    convert_dymaipp_staticbatch_om $resnet_onnx_file $SOC_VERSION "${staticbatch[*]}" $input_tensor_name $DYM_AIPPCONFIG_FILE_PATH || { echo "convert dymaipp static om failed";return 1; }

    dymbatch="1,2,4,8,16"
    convert_dymbatch_om $resnet_onnx_file $SOC_VERSION $dymbatch $input_tensor_name $AIPPCONFIG_FILE_PATH || { echo "convert dymbatch om failed";return 1; }
    dymbatch="1,2,4"
    convert_dymaipp_dymbatch_om $resnet_onnx_file $SOC_VERSION $dymbatch $input_tensor_name $DYM_AIPPCONFIG_FILE_PATH || { echo "convert dymaipp dymbatch om failed";return 1; }

    dymhw="224,224;448,448"
    unset AIPPCONFIG_FILE_PATH
    convert_dymhw_om $resnet_onnx_file $SOC_VERSION $dymhw $input_tensor_name $AIPPCONFIG_FILE_PATH || { echo "convert dymhw om failed";return 1; }
    dymhw="112,112;224,224"
    convert_dymaipp_dymhw_om $resnet_onnx_file $SOC_VERSION $dymhw $input_tensor_name $DYM_AIPPCONFIG_FILE_PATH || { echo "convert dymhw om failed";return 1; }

    dymdims="1,224,224;8,448,448"
    convert_dymdim_om $resnet_onnx_file $SOC_VERSION $dymdims $input_tensor_name $AIPPCONFIG_FILE_PATH || { echo "convert dymdim om failed";return 1; }

    convert_multi_dymaipp_om $two_dymaipp_onnx_file $SOC_VERSION $MULTI_DYM_AIPPCONFIG_FILE_PATH || { echo "convert multi dymaipp om failed";return 1; }


    # dymshapes 310 不支持，310P支持
    dymshapes="[1~16,3,200~300,200~300]"
    convert_dymshape_om $resnet_onnx_file $SOC_VERSION $dymshapes $input_tensor_name $AIPPCONFIG_FILE_PATH || { echo "convert dymshape om failed";return 1; }
    if [ ! -f $TESTDATA_PATH/pth_resnet50_dymshape.om ]; then
        mv $TESTDATA_PATH/pth_resnet50_dymshape*.om $TESTDATA_PATH/pth_resnet50_dymshape.om
    fi
}

main "$@"
exit $?
