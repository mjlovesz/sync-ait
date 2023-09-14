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
    cd $CUR_PATH/../analyze/tests/model_eval/
    bash test.sh
}
function test_benchmark()
{
    cd $CUR_PATH/../benchmark/test/
    bash test.sh $1 $2
}
function test_convert()
{
    cd $CUR_PATH/../convert/test/
    bash test.sh
}
function test_debug_compare()
{
    cd $CUR_PATH/../debug/compare/tests/ut/
    bash test.sh

    cd $CUR_PATH/../debug/compare/tests/st/
    bash test.sh
}
function test_debug_surgeon()
{
    cd $CUR_PATH/../debug/surgeon/test/
    bash test.sh
}
function test_profile()
{
    cd $CUR_PATH/../profile/test/
    bash test.sh
}
function test_transplt()
{
    cd $CUR_PATH/../transplt/test/
    bash test.sh
}

main() {
    export dt_mode=${1:-"normal"} # or "pr"
    dt_list=(0 0 0 0 0 0 0)
    if [[ $dt_mode == "pr" ]];then
        soft_link_path=/home/dcs-50/ait_test/ait/ait/components
        [[ -d $soft_link_path ]] || { echo "can't find origin dt data";return $ret_failed; }
        cur_testdata_path=$CUR_PATH/../benchmark/test/testdata
        [[ -d $cur_testdata_path ]] || { `ln -s $soft_link_path/benchmark/test/testdata $cur_testdata_path`; }
        modify_files=$CUR_PATH/../../../../modify_files.txt
        if [[ -f $modify_files ]];then
            echo "found modify_files"
            while read line
            do
                result=""
                result=$(echo $line | grep "components/analyze")
                [[ $result == "" ]] || { dt_list[0]=1;echo "run analyze DT"; }
                result=""
                result=$(echo $line | grep "components/benchmark")
                [[ $result == "" ]] || { dt_list[1]=1;echo "run benchmark DT"; }
                result=""
                result=$(echo $line | grep "components/convert")
                [[ $result == "" ]] || { dt_list[2]=1;echo "run convert DT"; }
                result=""
                result=$(echo $line | grep "components/debug/compare")
                [[ $result == "" ]] || { dt_list[3]=1;echo "run compare DT"; }
                result=""
                result=$(echo $line | grep "components/debug/surgeon")
                [[ $result == "" ]] || { dt_list[4]=1;echo "run surgeon DT"; }
                result=""
                result=$(echo $line | grep "components/profile")
                [[ $result == "" ]] || { dt_list[5]=1;echo "run profile DT"; }
                result=""
                result=$(echo $line | grep "components/transplt")
                [[ $result == "" ]] || { dt_list[6]=1;echo "run transplt DT"; }

            done < $modify_files
        fi
    else
        dt_list=(1 1 1 1 1 1 1)
    fi
    echo "dt_list ${dt_list[@]}"

    get_npu_type || { echo "invalid npu device";return $ret_failed; }
    PYTHON_COMMAND="python3"
    BENCKMARK_DT_MODE="simple"

    all_part_test_ok=$ret_ok
    if [[ ${dt_list[0]} -eq 1 ]];then
        test_analyze || { echo "developer test analyze failed";all_part_test_ok=$ret_failed; }
    fi
    if [[ ${dt_list[1]} -eq 1 ]];then
        test_benchmark $SOC_VERSION $PYTHON_COMMAND $BENCKMARK_DT_MODE || { echo "developer test benchmark failed";all_part_test_ok=$ret_failed; }
    fi
    if [[ ${dt_list[2]} -eq 1 ]];then
        test_convert || { echo "developer test convert failed";all_part_test_ok=$ret_failed; }
    fi
    if [[ ${dt_list[3]} -eq 1 ]];then
        test_debug_compare || { echo "developer test comnpare failed";all_part_test_ok=$ret_failed; }
    fi
    if [[ ${dt_list[4]} -eq 1 ]];then
        test_debug_surgeon || { echo "developer test surgeon failed";all_part_test_ok=$ret_failed; }
    fi
    if [[ ${dt_list[5]} -eq 1 ]];then
        test_profile || { echo "developer test profile failed";all_part_test_ok=$ret_failed; }
    fi
    if [[ ${dt_list[6]} -eq 1 ]];then
        test_transplt || { echo "developer test transplt failed";all_part_test_ok=$ret_failed; }
    fi
    cd $CUR_PATH

    return $all_part_test_ok
}

main "$@"
exit $?