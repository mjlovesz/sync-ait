# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
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

set -u

pwd_dir=${PWD}

# copy auto_optimizer to test file, and test
cp ${pwd_dir}/../auto_optimizer ${pwd_dir}/ -rf

coverage run -p -m unittest
ret=$?
if [ $ret != 0 ]; then
    echo "coverage run failed! "
    exit -1
fi

coverage combine
coverage report -m --omit="test_*.py" > ${pwd_dir}/test.coverage

coverage_line=`cat ${pwd_dir}/test.coverage | grep "TOTAL" | awk '{print $4}' | awk '{print int($0)}'`

target=60
if [ ${coverage_line} -lt ${target} ]; then
    echo "coverage failed! coverage_line=${coverage_line}, Coverage does not achieve target(${target}%), Please add ut case."
    exit -1
fi

echo "coverage_line=${coverage_line}"
