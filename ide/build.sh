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

set -e

script=$(readlink -f "$0")
route=$(dirname "$script")

install_path=${route}/workspace/install
download_path=${route}/workspace/3rd_party

build_type=$1
product_type=$2
beta=$3

function build_plugins() {
  bash ${route}/build_ide.sh || exit 2
}

function pack() {
  cd ${download_path}
  cd ${download_path}/intellij-community/intellij-community
  rm -rf ${download_path}/intellij-community/patch/*
  cp -rf ${route}/../3rd-party-tools/patch/intellij-community/* ${download_path}/intellij-community/patch/

  source ${route}/version
  bash build.sh $version_major $version_minor $edition_version $package_name
}

function main() {
  if [ "${build_type}" == "build" ]; then
    echo "run build mission"
    build_plugins
  elif [ "${build_type}" == "install" ]; then
    build_plugins
    bash ${route}/install.sh $1 $2 $3 || exit 9
    pack
  else
    echo "input args is error" || exit 1
  fi
}

main