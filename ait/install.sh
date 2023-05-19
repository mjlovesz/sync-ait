#!/usr/bin/env bash

CURRENT_DIR=$(dirname $(readlink -f $0))
arg_force_reinstall=
only_debug=
only_profile=
only_analyze=
only_transplt=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -f) arg_force_reinstall=--force-reinstall;;
  --debug) only_debug=true;;
  --profile) only_profile=true;;
  --analyze) only_analyze=true;;
  --transplt) only_transplt=true;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

python_version=$(python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}')

if [ $python_version -eq 2 ]
then
  echo "Your python version is 2" >&2
  exit 1;
elif [ $python_version -eq 3 ]
then
  echo "Your python version is 3"
fi

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h      : Print help menu"
  echo " --debug : only install debug component"
  echo " --profile : only install profile component"
  echo " --analyze : only install analyze component"
  echo " --transplt : only install transplt component"
  exit;
fi

if [ ! "$(command -v pip)" ]; then
  echo "pip 没有安装" >&2
  exit 1;
fi

pip install ${CURRENT_DIR} ${arg_force_reinstall}

if [ ! -z $only_debug ]
then
  pip install ${CURRENT_DIR}/components/debug/compare \
  ${CURRENT_DIR}/components/debug/surgeon \
  ${arg_force_reinstall}
fi

if [ ! -z $only_profile ]
then
  pip install ${CURRENT_DIR}/components/profile/benchmark/backend \
  ${CURRENT_DIR}/components/profile/benchmark \
  ${arg_force_reinstall}
fi

if [ ! -z $only_analyze ]
then
  pip install ${CURRENT_DIR}/components/analyze \
  ${arg_force_reinstall}
fi

if [ ! -z $only_transplt ]
then
  pip install ${CURRENT_DIR}/components/transplt \
  ${arg_force_reinstall}
fi

if [ -z $only_debug ] && [ -z $only_profile ] && [ -z $only_analyze ] && [ -z $only_transplt ]
then
  pip install ${CURRENT_DIR}/components/debug/compare \
  ${CURRENT_DIR}/components/debug/surgeon \
  ${CURRENT_DIR}/components/profile/benchmark/backend \
  ${CURRENT_DIR}/components/profile/benchmark \
  ${CURRENT_DIR}/components/analyze \
  ${CURRENT_DIR}/components/transplt \
  ${arg_force_reinstall}
fi
