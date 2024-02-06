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

function get_modified_module_list() {
    dt_list=(0 0 0 0 0 0 0 0)
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
            result=""
            result=$(echo $line | grep "components/llm")
            [[ $result == "" ]] || { dt_list[7]=1;echo "run llm DT"; }
        done < $modify_files
    fi
}

main() {
    export dt_mode=${1:-"normal"} # or "pr"
    if [[ $dt_mode == "pr" ]];then
        dt_list=get_modified_module_list
    else
        dt_list=(1 1 1 1 1 1 1)
    fi
    echo "dt_list ${dt_list[@]}"

    failed_case_names=""
    all_part_test_ok=0
    TEST_CASES=( $(find ./* -name test.sh) )
    for test_case in ${TEST_CASES[@]}; do
        echo ">>>> Current test_case=$test_case"
        CASE_PATH=`dirname $test_case`

        cd $CASE_PATH
        bash test.sh
        cur_result=$?
        echo ">>>> test_case=$test_case, cur_result=$cur_result"
        if [ "$cur_result" -eq "0" ]; then
            failed_case_names="$failed_case_names, $test_case"
            all_part_test_ok=$(( $all_part_test_ok + $cur_result ))
        fi
        cd $CUR_PATH
    done

    echo "failed_case_names: ${failed_case_names:2:}"  # Exclude the first ", "
    return $all_part_test_ok
}

main "$@"
exit $?