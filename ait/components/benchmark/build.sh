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
CURDIR=$(dirname $(readlink -f $0))
PLATFORM=`uname -i`
PACKET_NAME="ais_bench_$PLATFORM"
OUTPUT_PATH="$CURDIR/$PACKET_NAME/"

declare -i ret_ok=0
declare -i ret_failed=1


file_is_legal()
{
    path=$1
    owner=$(stat -c '%U' $path)
    group=$(stat -c '%G' $path)
    cur_user=$(id -un)
    cur_group=$(id -gn)
    if [[ "$owner" == "$cur_user" || "$group" == "$cur_group" ]];then
        echo "$path owner check passed"
    else
        echo "$path no belong to cur_user or cur_group"
        return $ret_failed
    fi
    if [[ -L "$path" ]];then
        echo "$path no belong to cur_user or cur_group"
        return $ret_failed
    else
        echo "$path owner check passed"
    fi
    return $ret_ok
}


safe_remove()
{
    path=$1
    if [[ -d $path ]];then
        check_ok=$ret_ok
        file_is_legal $path || { check_ok=$ret_failed; }
        if [[ $check_ok == $ret_ok ]];then
            rm -rf $path
            return $ret_ok
        else
            echo "not allowed to remove $path"
            return $ret_failed
        fi
    fi
    return $ret_ok
}


safe_remove_pattern()
{
    pattern=$1
    for file in $pattern; do
        check_ok=$ret_ok
        file_is_legal $path || { check_ok=$ret_failed; }
        if [[ $check_ok == $ret_ok ]];then
            rm -rf $file
        fi
    done
}


safe_cp()
{
    path=$1
    target_dir=$2
    rm_flag=$3 # remove origin file or not
    file_is_legal $target_dir || { echo "cp target $path is illegal";return $ret_failed; }
    if [[ -d $path ]];then
        file_is_legal $path || { echo "cp origin path $path is illegal";return $ret_failed; }
        if [[ $rm_flag == "true" ]];then
            cp $path -rf $target_dir

        else
            cp $path $target_dir
        fi
        chmod -R 750 $target_dir
        return $ret_ok
    fi
    return $ret_failed
}


safe_pattern_cp()
{
    pattern=$1
    target_dir=$2
    rm_flag=$3 # remove origin file or not
    file_is_legal $target_dir || { echo "cp target $path is illegal";return $ret_failed; }
    for file in $pattern; do
        check_ok=$ret_ok
        file_is_legal $path || { check_ok=$ret_failed; }
        if [[ $check_ok == $ret_ok ]];then
            if [[ $rm_flag == "true" ]];then
                cp $file -rf $target_dir
            else
                cp $file $target_dir
            fi
            chmod -R 750 $target_dir
        fi
        return $ret_ok
    done
}


main()
{
    safe_remove $OUTPUT_PATH || { echo "$OUT_PATH rm failed";return $ret_failed; }
    mkdir -p -m 750 $OUTPUT_PATH
    safe_remove_pattern $CURDIR/ais_bench*.whl
    safe_remove_pattern $CURDIR/aclruntime*.whl

    cd $CURDIR

    safe_remove_pattern $CURDIR/backend/*.egg-info
    safe_remove $CURDIR/backend/build
    which pip3.7 && { pip3.7 wheel -v $CURDIR/backend || echo "pip3.7 run failed"; }
    which pip3.8 && { pip3.8 wheel -v $CURDIR/backend || echo "pip3.8 run failed"; }
    which pip3.9 && { pip3.9 wheel -v $CURDIR/backend || echo "pip3.9 run failed"; }

    safe_remove_pattern $CURDIR/*.egg-info
    safe_remove $CURDIR/build
    which pip3.7 && { pip3.7 wheel -v $CURDIR || echo "pip3.7 run failed"; }
    which pip3.8 && { pip3.8 wheel -v $CURDIR || echo "pip3.8 run failed"; }
    which pip3.9 && { pip3.9 wheel -v $CURDIR || echo "pip3.9 run failed"; }

    safe_pattern_cp $CURDIR/aclruntime*.whl $OUTPUT_PATH "true" || { echo "$CURDIR/aclruntime*.whl cp failed";return $ret_failed; }
    safe_pattern_cp $CURDIR/ais_bench*.whl $OUTPUT_PATH "true" || { echo "$CURDIR/ais_bench*.whl cp failed";return $ret_failed; }

    safe_cp $CURDIR/ais_bench $OUTPUT_PATH "true" || { echo "$CURDIR/ais_bench cp failed";return $ret_failed; }
    safe_cp $CURDIR/requirements.txt $OUTPUT_PATH "false" || { echo "$CURDIR/requirements.txt cp failed";return $ret_failed; }
    safe_cp $CURDIR/README.md $OUTPUT_PATH "false" || { echo "$CURDIR/README.md cp failed";return $ret_failed; }
    safe_cp $CURDIR/FAQ.md $OUTPUT_PATH "false" || { echo "$CURDIR/FAQ.md cp failed";return $ret_failed; }

    cd $CURDIR
    safe_remove $CURDIR/$PACKET_NAME.tar.gz || { return $ret_failed; }
    tar -czf $CURDIR/$PACKET_NAME.tar.gz $PACKET_NAME

    safe_remove ${CURDIR}/output || { return $ret_failed; }

    mkdir -p -m 750 ${CURDIR}/output
    safe_pattern_cp $CURDIR/$PACKET_NAME.tar.gz ${CURDIR}/output "false"
}

main "$@"
exit $?
