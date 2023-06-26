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

install_clang() {
  local SUDO=""
  type sudo
  ret=$?
  if [ $ret -eq 0 ]; then
    SUDO="sudo"
  fi

  $SUDO apt-get install wget unzip -y

  os_name=$(grep -E "^NAME=" /etc/os-release | cut -d'=' -f2 | tr -d '"')
  os_version=$(grep -E "^VERSION=" /etc/os-release | cut -d'=' -f2 | cut -d' ' -f1 | sed 's/\"//g')

  if [ "$os_name" == "Ubuntu" ]; then
    if [ -n "${os_version%%22.04*}" ] || [ -n "${os_version%%20.04*}" ]; then
      $SUDO apt-get install libclang-14-dev clang-14 -y
    elif [ -n "${os_version%%18.04*}" ]; then
      $SUDO apt-get install libclang-10-dev clang-10 -y
    fi
  elif [ -n "${os_name%%CentOS*}" ] && [ -n "${os_version%%7*}" ]; then
    yum install centos-release-scl-rh -y
    yum install llvm-toolset-7.0-clang -y
    source /opt/rh/llvm-toolset-7.0/enable
    echo "source /opt/rh/devtoolset-7/enable" >> ~/.bashrc
    export CPLUS_INCLUDE_PATH=/usr/local/lib/clang/7.0.0/include:$CPLUS_INCLUDE_PATH
    echo "export CPLUS_INCLUDE_PATH=/usr/local/lib/clang/7.0.0/include:\$CPLUS_INCLUDE_PATH" >> ~/.bashrc
  elif [ "$os_name" == "SLES" ] && [ -n "${os_version%%12*}" ]; then
    $SUDO zypper install libclang7 clang7-devel -y
  else
    echo "WARNING: uncertified os type:version $os_name:$os_version. Ait transplt installation may be incorrect!!!"
    $SUDO apt-get install libclang-14-dev clang-14 -y
  fi
}


# Download and unzip config.zip, headers.zip
download_config_and_headers() {
  cd $(python3 -c "import app_analyze; print(app_analyze.__path__[0])") \
    && wget -O config.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/config.zip \
    && unzip -o -q config.zip \
    && rm config.zip \
    && wget -O headers.zip https://ait-resources.obs.cn-south-1.myhuaweicloud.com/headers.zip \
    && unzip -o -q headers.zip \
    && rm headers.zip
}

# Install clang
install_clang

download_config_and_headers
