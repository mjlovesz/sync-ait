
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

OM_MODEL_XX_PATH = $CUR_PATH/om/model_xx.om
ONNX_MODEL_XX_PATH = $CUR_PATH/onnx/model_xx.om

function get_om_model_xx()
{
    local get_model="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $convert_url  -O $1 --no-check-certificate
}

function get_onnx_model_xx()
{
    local get_model="https://aisbench.obs.myhuaweicloud.com/packet/msame/x86/msame" # fake url
    wget $convert_url  -O $1 --no-check-certificate
}

get_om_model_xx $OM_MODEL_XX_PATH || { echo "get OM_MODEL_XX failed";return $ret_failed; }
get_onnx_model_xx $ONNX_MODEL_XX_PATH || { echo "get ONNX_MODEL_XX failed";return $ret_failed; }

