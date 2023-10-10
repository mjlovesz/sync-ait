declare -i ret_ok=0
declare -i ret_failed=1
SOC_VERSION=""
CUR_PATH=$(dirname $(readlink -f "$0"))
MODEL_PATH="$CUR_PATH/testdata/add_model/model"
ONNX_PATH="$MODEL_PATH/add_model.onnx"

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

# 静态batch转换
convert_static_om()
{
    local _om_path_pre="$MODEL_PATH/add_model_bs1"
    local _om_path="$_om_path_pre.om"
    local _input_shape="input1:1,3,32,32;input2:1,3,32,32"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$ONNX_PATH --output=$_om_path_pre --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION"
        $_cmd || { echo "static model atc run failed";return $ret_failed; }
    fi
}

# 动态batch转换
convert_dymbatch_om()
{
    local _om_path_pre="$MODEL_PATH/add_model_dymbatch"
    local _om_path="$_om_path_pre.om"
    local _input_shape="input1:-1,3,32,32;input2:-1,3,32,32"
    local _dymbatch="1,2,4,8"

    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$ONNX_PATH --output=$_om_path_pre --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION --dynamic_batch_size=$_dymbatch"
        $_cmd || { echo "dymbatch model atc run failed";return $ret_failed; }
    fi
}

# 动态分辨率转换
convert_dymwh_om()
{
    local _om_path_pre="$MODEL_PATH/add_model_dymwh"
    local _om_path="$_om_path_pre.om"
    local _input_shape="input1:1,3,-1,-1;input2:1,3,-1,-1"
    local _dymhw="32,32;64,64"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$ONNX_PATH --output=$_om_path_pre --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION --dynamic_image_size=$_dymhw"
        $_cmd || { echo "dymwh model atc run failed";return $ret_failed; }
    fi
}

# 动态dims转换
convert_dymdim_om()
{
    local _om_path_pre="$MODEL_PATH/add_model_dymdim"
    local _om_path="$_om_path_pre.om"
    local _input_shape="input1:-1,3,-1,-1;input2:-1,3,-1,-1"
    local _dymdim="1,32,32,1,32,32;4,64,64,4,64,64"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$ONNX_PATH --output=$_om_path_pre --input_format=ND --framework=5 \
            --input_shape=$_input_shape --soc_version=$SOC_VERSION --dynamic_dims=$_dymdim"
        $_cmd || { echo "dymwh model atc run failed";return $ret_failed; }
    fi
}


convert_dymshape_om()
{
    local _om_path_pre="$MODEL_PATH/add_model_dymshape"
    local _om_path="$_om_path_pre.om"
    local _input_shape="input1:[1~4,3,32~64,32~64];input2:[1~4,3,32~64,32~64]"
    if [[ ! -f $_om_path ]]; then
        local _cmd="atc --model=$ONNX_PATH --output=$_om_path_pre --framework=5 \
            --input_shape_range=$_input_shape --soc_version=$SOC_VERSION"
        $_cmd || { echo "dymwh model atc run failed";return $ret_failed; }
    fi
}

main()
{
    [ -d $MODEL_PATH ] || mkdir -p $MODEL_PATH
    if [[ ! -f $ONNX_PATH ]]; then
        python3 generate_add_model.py || { echo "generate add onnx failed";return $ret_failed; }
        mv $CUR_PATH/add_model.onnx $MODEL_PATH/ || { echo "move onnx failed";return $ret_failed; }
    fi

    get_npu_type || { echo "get npu type failed";return $ret_failed; }
    convert_static_om || { return $ret_failed; }
    convert_dymbatch_om || { return $ret_failed; }
    convert_dymwh_om || { return $ret_failed; }
    convert_dymdim_om || { return $ret_failed; }
    if [ $SOC_VERSION != "Ascend310" ]; then
        convert_dymshape_om || { return $ret_failed; }
        if [ ! -f $MODEL_PATH/add_model_dymshape.om ]; then
            mv $MODEL_PATH/add_model_dymshape*.om $MODEL_PATH/add_model_dymshape.om
        fi
    fi

    return $ret_ok
}
main "$@"
exit $?

