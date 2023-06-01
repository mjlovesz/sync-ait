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

OM_MODEL_DATA2VEC_PATH = $CUR_PATH/om/data2vec_1_108.om
ONNX_MODEL_DATA2VEC_PATH = $CUR_PATH/onnx/data2vec_1_108.onnx
INPUT_DATA_DATA2VEC_PATH = $CUR_PATH/input_datas/data2vec/

OM_MODEL_GELU_PATH = $CUR_PATH/om/gelu.om
ONNX_MODEL_GELU_PATH = $CUR_PATH/onnx/gelu.onnx
INPUT_DATA_GELU_PATH = $CUR_PATH/input_datas/gelu/

function get_om_model_data2vec()
{
    local get_om="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_om  -O $1 --no-check-certificate
}

function get_onnx_model_data2vec()
{
    local get_onnx="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_onnx  -O $1 --no-check-certificate
}

function get_input_data_data2vec()
{
    local get_input="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_input  -O $1 --no-check-certificate
    local get_input="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_input  -O $1 --no-check-certificate
}

function get_om_model_gelu()
{
    local get_om="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_om  -O $1 --no-check-certificate
}

function get_onnx_model_gelu()
{
    local get_onnx="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_onnx  -O $1 --no-check-certificate
}

function get_input_data_gelu()
{
    local get_input="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $get_input  -O $1 --no-check-certificate
}


get_om_model_data2vec $OM_MODEL_DATA2VEC_PATH || { echo "get OM_MODEL_DATA2VEC failed";return $ret_failed; }
get_onnx_model_data2vec $ONNX_MODEL_DATA2VEC_PATH || { echo "get ONNX_MODEL_DATA2VEC failed";return $ret_failed; }
get_input_data_data2vec $INPUT_DATA_DATA2VEC_PATH || { echo "get INPUT_DATA_DATA2VEC failed";return $ret_failed; }

get_om_model_gelu $OM_MODEL_GELU_PATH || { echo "get OM_MODEL_GELU failed";return $ret_failed; }
get_onnx_model_gelu $ONNX_MODEL_GELU_PATH || { echo "get ONNX_MODEL_GELU failed";return $ret_failed; }
get_input_data_gelu $INPUT_DATA_GELU_PATH || { echo "get INPUT_DATA_GELU failed";return $ret_failed; }
