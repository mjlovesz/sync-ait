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
CUR_PATH=$("pwd")
SOC_VERSION=""

function get_npu_type()
{
    get_npu_310=`lspci | grep d100`
    get_npu_310P3=`lspci | grep d500`
    get_npu_310B=`lspci | grep d107`
    if [[ $get_npu_310 != "" ]];then
        SOC_VERSION="Ascend310"
        echo "npu is Ascend310"
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

function test_analyze()
{
    bash $CUR_PATH/../analyze/tests/test.sh
}
function test_benchmark()
{
    bash $CUR_PATH/../benchmark/test/test.sh $1 $2
}
function test_convert()
{
    bash $CUR_PATH/../convert/test/test.sh
}
function test_debug_compare()
{
    bash $CUR_PATH/../debug/compare/tests/ut/test.sh
}
function test_debug_surgeon()
{
    bash $CUR_PATH/../debug/surgeon/test/test.sh
}
function test_profile()
{
    bash $CUR_PATH/../profile/test/test.sh
}
function test_transplt()
{
    bash $CUR_PATH/../transplt/test/test.sh
}

main() {
    get_npu_type || { echo "invalid npu device";return $ret_failed; }
    PYTHON_COMMAND="python3"
    BENCKMARK_DT_MODE="simple"

    test_analyze || { echo "developer test analyze failed";return $ret_failed; }
    test_benchmark $SOC_VERSION $PYTHON_COMMAND $BENCKMARK_DT_MODE || { echo "developer test benchmark failed";return $ret_failed; }
    test_convert || { echo "developer test convert failed";return $ret_failed; }
    test_debug_compare || { echo "developer test comnpare failed";return $ret_failed; }
    test_debug_surgeon || { echo "developer test surgeon failed";return $ret_failed; }
    test_profile || { echo "developer test profile failed";return $ret_failed; }
    test_transplt || { echo "developer test transplt failed";return $ret_failed; }

    return $ret_ok
}

main "$@"
exit $?