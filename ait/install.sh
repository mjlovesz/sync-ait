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
arg_force_reinstall=
only_debug=
only_benchmark=
only_analyze=
only_convert=
only_transplt=
only_profile=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -f) arg_force_reinstall=--force-reinstall;;
  --full) full_install=--full;;
  --debug) only_debug=true;;
  --benchmark) only_benchmark=true;;
  --analyze) only_analyze=true;;
  --convert) only_convert=true;;
  --transplt) only_transplt=true;;
  --profile) only_profile=true;;
  --uninstall) uninstall=true;;
  -y) all_uninstall=-y;;
  -i) shift
    pip_source_url=$1
    curl_ans=`curl ${pip_source_url} -s --head | head -n 1 | grep "HTTP/1.[01] [23].."`
    if [ -z "${curl_ans}" ];then
      echo "Error: Invalid pip source: ${pip_source_url}"
      exit 1;
    fi
    echo "Info: Using pip source ${pip_source_url}"
    pip_source="-i ${pip_source_url}";;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

if [ ! "$(command -v python3)" ]
then
  echo "Error: python3 is not installed" >&2
  exit 1;
fi

if [ ! "$(command -v pip3)" ]; then
  echo "Error: pip3 is not installed" >&2
  exit 1;
fi

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h : Print help menu"
  echo " --debug : only install debug component"
  echo " --benchmark : only install benchmark component"
  echo " --analyze : only install analyze component"
  echo " --convert : only install convert component"
  echo " --transplt : only install transplt component"
  echo " --profile : only install profile component"
  echo " --full : using with install, install all components and dependencies, may need sudo privileges"
  echo " --uninstall : uninstall"
  echo " -y : using with uninstall, don't ask for confirmation of uninstall deletions"
  exit;
fi


uninstall(){
  if [ -z $only_debug ] && [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_convert ] && [ -z $only_transplt ] && [ -z $only_profile ]
  then
    pip3 uninstall ait analyze_tool aclruntime ais_bench convert_tool compare auto_optimizer msprof transplt ${all_uninstall}
  else
    if [ ! -z $only_debug ]
    then
      pip3 uninstall compare auto_optimizer ${all_uninstall}
    fi

    if [ ! -z $only_benchmark ]
    then
      pip3 uninstall aclruntime ais_bench ${all_uninstall}
    fi

    if [ ! -z $only_analyze ]
    then
      pip3 uninstall analyze_tool ${all_uninstall}
    fi

    if [ ! -z $only_convert ]
    then
      pip3 uninstall convert_tool ${all_uninstall}
    fi

    if [ ! -z $only_transplt ]
    then
      pip3 uninstall transplt ${all_uninstall}
    fi

    if [ ! -z $only_profile ]
    then
      pip3 uninstall msprof ${all_uninstall}
    fi
  fi
  exit;
}


install(){
  pip3 install ${CURRENT_DIR} ${arg_force_reinstall} ${pip_source}

  if [ ! -z $only_debug ]
  then
    pip3 install ${CURRENT_DIR}/components/debug/compare \
    ${CURRENT_DIR}/components/debug/surgeon \
    ${arg_force_reinstall} ${pip_source}
  fi

  if [ ! -z $only_benchmark ]
  then
    pip3 install ${CURRENT_DIR}/components/benchmark/backend \
    ${CURRENT_DIR}/components/benchmark \
    ${arg_force_reinstall} ${pip_source}
  fi

  if [ ! -z $only_analyze ]
  then
    pip3 install ${CURRENT_DIR}/components/analyze \
    ${arg_force_reinstall} ${pip_source}
  fi

  if [ ! -z $only_convert ]
  then
    pip3 install ${CURRENT_DIR}/components/convert \
    ${arg_force_reinstall} ${pip_source}

    bash ${CURRENT_DIR}/components/convert/build.sh
  fi

  if [ ! -z $only_transplt ]
  then
    pip3 install ${CURRENT_DIR}/components/transplt \
    ${arg_force_reinstall} ${pip_source}
    source ${CURRENT_DIR}/components/transplt/install.sh $full_install
  fi

  if [ ! -z $only_profile ]
  then
    pip3 install ${CURRENT_DIR}/components/profile/msprof \
    ${arg_force_reinstall} ${pip_source}
  fi

  if [ -z $only_debug ] && [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_convert ] && [ -z $only_transplt ] && [ -z $only_profile ]
  then
    pip3 install ${CURRENT_DIR}/components/debug/compare \
    ${CURRENT_DIR}/components/debug/surgeon \
    ${CURRENT_DIR}/components/benchmark/backend \
    ${CURRENT_DIR}/components/benchmark \
    ${CURRENT_DIR}/components/analyze \
    ${CURRENT_DIR}/components/convert \
    ${CURRENT_DIR}/components/transplt \
    ${CURRENT_DIR}/components/profile/msprof \
    ${arg_force_reinstall} ${pip_source}

    if [ ! ${AIE_DIR} ];then
      echo "Warning: Ascend Inference Engine is not installed. (convert install failed)"
    else
      bash ${CURRENT_DIR}/components/convert/build.sh
    fi

    source ${CURRENT_DIR}/components/transplt/install.sh $full_install
  fi

  rm -rf ${CURRENT_DIR}/ait.egg-info
}


if [ ! -z $uninstall ]
then
  uninstall
else
  install
fi