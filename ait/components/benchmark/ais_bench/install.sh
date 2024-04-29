#!/usr/bin/env bash
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
CURRENT_DIR=$(dirname $(readlink -f $0))


download_and_install_aclruntime() {
    ACLRUNTIME_VERSION=`pip3 show aclruntime | awk '/Version: /{print $2}'`

    if [ "$ACLRUNTIME_VERSION" = "0.0.2" ]; then
        echo "aclruntime==0.0.2 already installed, skip"
        return
    fi

    echo "download and install aclruntime"
    PYTHON3_MINI_VERSION=`python3 --version | cut -d'.' -f 2`
    if [ "$PYTHON3_MINI_VERSION" = "7" ]; then
        SUB_SUFFIX="m"
    else
        SUB_SUFFIX=""
    fi
    echo "PYTHON3_MINI_VERSION=$PYTHON3_MINI_VERSION, SUB_SUFFIX=$SUB_SUFFIX"
    WHL_NAME="aclruntime-0.0.2-cp3${PYTHON3_MINI_VERSION}-cp3${PYTHON3_MINI_VERSION}${SUB_SUFFIX}-linux_$(uname -m).whl"
    BASE_URL="https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/"
    echo "WHL_NAME=$WHL_NAME, URL=${BASE_URL}${WHL_NAME}"
    wget --no-check-certificate -c "${BASE_URL}${WHL_NAME}" && pip3 install $WHL_NAME --force-reinstall && rm -f $WHL_NAME
    if [ $? -ne 0 ]; then
        echo "Downloading or installing from whl failed, will install from source code"
        cd ${CURRENT_DIR}/../backend && pip install . --force-reinstall && cd -
    fi
}

download_and_install_aclruntime
