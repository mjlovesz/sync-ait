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

CUR_PATH=$(dirname $(readlink -f $0))
COMPONENTS_PATH=$CUR_PATH/../../../
SOURCE_CODE_PATH=$COMPONENTS_PATH/components/utils
echo "CUR_PATH=$CUR_PATH, COMPONENTS_PATH=$COMPONENTS_PATH, SOURCE_CODE_PATH=$SOURCE_CODE_PATH"

if [ -f "../requirements.txt" ]; then
    pip3 install -r ../requirements.txt
fi

if [ -f "$CUR_PATH/resources" ]; then
    chmod -R 750 $CUR_PATH/resources
fi

PYTHONPATH=$COMPONENTS_PATH:$PYTHONPATH coverage run --source $SOURCE_CODE_PATH coverage run -m pytest -vv $CUR_PATH --disable-warnings

RETURN_CODE=0
if [ $? == 0 ]; then
    coverage combine $CUR_PATH
    coverage report -m --omit="test_*.py" -i > $CUR_PATH/test.coverage
    coverage_rate=$(awk '/TOTAL/{print $4}' $CUR_PATH/test.coverage | cut -d '%' -f 1)
    echo "coverage_rate=$coverage_rate%"

    coverage_target=50  # Current is only 51%
    if [[ "$coverage_rate" -ne "" && "$coverage_rate" -lt "$target" ]]; then
        echo "coverage rate too low(<${coverage_target}%), currently reaches only ${coverage_rate}%."
        RETURN_CODE=1
    fi
else
    echo "coverage run failed! "
    RETURN_CODE=1
fi

unlink $COMPONENTS_PATH
exit $RETURN_CODE
