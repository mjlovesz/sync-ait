#!/usr/bin/env bash

CURRENT_DIR=$(dirname $(readlink -f $0))
arg_force_reinstall=
only_debug=
only_benchmark=
only_analyze=
only_transplt=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -f) arg_force_reinstall=--force-reinstall;;
  --debug) only_debug=true;;
  --benchmark) only_benchmark=true;;
  --analyze) only_analyze=true;;
  --transplt) only_transplt=true;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

python_version=$(python3 -V)

if [ -z $python_version ]
then
  echo "ERROR: python3 -V shows that you do not have python3" >&2
  exit 1;
fi

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h      : Print help menu"
  echo " --debug : only install debug component"
  echo " --benchmark : only install benchmark component"
  echo " --analyze : only install analyze component"
  echo " --transplt : only install transplt component"
  exit;
fi

if [ ! "$(command -v pip3)" ]; then
  echo "pip3 没有安装" >&2
  exit 1;
fi

pip3 install ${CURRENT_DIR} ${arg_force_reinstall}

if [ ! -z $only_debug ]
then
  pip3 install ${CURRENT_DIR}/components/debug/compare \
  ${CURRENT_DIR}/components/debug/surgeon \
  ${arg_force_reinstall}
fi

if [ ! -z $only_benchmark ]
then
  pip3 install ${CURRENT_DIR}/components/benchmark/backend \
  ${CURRENT_DIR}/components/benchmark \
  ${arg_force_reinstall}
fi

if [ ! -z $only_analyze ]
then
  pip3 install ${CURRENT_DIR}/components/analyze \
  ${arg_force_reinstall}
fi

if [ ! -z $only_transplt ]
then
  pip3 install ${CURRENT_DIR}/components/transplt \
  ${arg_force_reinstall}
fi

if [ -z $only_debug ] && [ -z $only_benchmark ] && [ -z $only_analyze ] && [ -z $only_transplt ]
then
  pip3 install ${CURRENT_DIR}/components/debug/compare \
  ${CURRENT_DIR}/components/debug/surgeon \
  ${CURRENT_DIR}/components/benchmark/backend \
  ${CURRENT_DIR}/components/benchmark \
  ${CURRENT_DIR}/components/analyze \
  ${CURRENT_DIR}/components/transplt \
  ${arg_force_reinstall}
fi
