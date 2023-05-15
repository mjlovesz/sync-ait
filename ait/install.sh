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

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h      : Print help menu"
  echo " --force-reinstall : reinstall"
  exit;
fi

python3 -m pip install ${CURRENT_DIR} \
${CURRENT_DIR}/components/debug/compare \
${CURRENT_DIR}/components/debug/surgeon \
${CURRENT_DIR}/components/profile/benchmark/backend \
${CURRENT_DIR}/components/profile/benchmark \
${arg_force_reinstall}

