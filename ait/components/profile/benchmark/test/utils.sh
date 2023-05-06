#!/bin/bash

# Copyright 2022 Huawei Technologies Co., Ltd. All rights reserved.
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

check_python_package_is_install() {
    local PYTHON_COMMAND=$1
    ${PYTHON_COMMAND} -c "import $2" >>/dev/null 2>&1
    ret=$?
    if [ $ret != 0 ]; then
        echo "python package:$1 not install"
        return 1
    fi
    return 0
}
