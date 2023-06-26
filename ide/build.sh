#!/usr/bin/env bash
#Copyright 2023 Huawei Technologies Co., Ltd
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
# ================================================================================
set -e

intellij_version=$1
compile_only=N

if [[ "" != "$IDE_COMPILE_ONLY" ]]; then
  compile_only=$IDE_COMPILE_ONLY
fi

if [[ "" == "$intellij_version" ]]; then
  intellij_version="2022.3.2"
fi

if [[ "" == "$X_TEST" ]]; then
  x_test=$X_TEST
fi

function init() {
    script=$(readlink -f "$0")
    route=$(dirname "$script")
}

function build_ascend_plugin() {
  local ascend_plugin_name=$1
  local plugin_build_task=$2
  local not_exit_when_failed=$3
  echo "build plugin $plugin_name $(date +%Y-%m-%d_%H:%M:%S) task:$plugin_build_task"
  cd ${route}
  if [ 0"$GRADLE_URL" = "0" ]; then
     gradle wrapper
  else
     gradle wrapper --gradle-distributed-url $GRADLE_URL
  fi
  chmod a+x gradlew
  ./gradlew clean || {
    echo "gradlew clean ascend $plugin_name failed"
    exit 2
  }
  if [[ "$x_test" = "Y" ]] && [[ "plugin_build_task" = "build" ]]; then
    ./gradlew build -x test || {
      echo "build ascend $plugin_name failed"
      exit 3
    }
  else
    ./gradlew $plugin_build_task || {
      echo "build ascend $plugin_name failed"
      exit 4
  }
  fi
}

function clean() {
    rm -rf ~/.gradle/caches/modules-2/files-2.1/com.jetbrains.intellij.idea/unzipped.com.jetbrains.plugins
    if [ -f $plugins_path/builtinRegistry*.xml ]; then
       rm -rf ${plugins_path}/builtinRegistry*.xml
    fi
}

function main() {
  init
  clean

  local plugin_build_task=build
  if [[ "${compile_only}" != "Y" ]]; then
    plugin_build_task=buildPlugin
  fi
  clean && build_ascend_plugin ide $plugin_build_task
}

main
