#!/usr/bin/env bash

CURRENT_DIR=$(dirname $(readlink -f $0))
arg_force_reinstall=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -f) arg_force_reinstall=--force-reinstall;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

python_version=$(python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}')

if [ $python_version -eq 2 ]
then
  echo "Your python version is 2" >&2
elif [ $python_version -eq 3 ]
then
  echo "Your python version is 3"
fi

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h      : Print help menu"
  echo " --force-reinstall : reinstall"
  exit;
fi

if [ ! "$(command -v pip)" ]; then
  echo "pip 没有安装" >&2
  exit 1;
fi

pip install ${CURRENT_DIR} \
${CURRENT_DIR}/components/debug/compare \
${CURRENT_DIR}/components/debug/surgeon \
${CURRENT_DIR}/components/benchmark/backend \
${CURRENT_DIR}/components/benchmark \
${arg_force_reinstall}

