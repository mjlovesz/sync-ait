#!/usr/bin/env bash

CURRENT_DIR=$(dirname $(readlink -f $0))
arg_force_reinstall=
arg_help=0

while [[ "$#" -gt 0 ]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";exit 1;
esac; shift; done

if [ "$arg_help" -eq "1" ]; then
  echo "Usage: $0 [options]"
  echo " --help or -h      : Print help menu"
  echo " --force-reinstall : reinstall"
  exit;
fi

if [ ! -z ${arg_force_reinstall} ]; then
  pip uninstall ait compare aclruntime ais_bench
fi

pip install ${CURRENT_DIR}

pip install ${CURRENT_DIR}/components/debug/compare

pip install ${CURRENT_DIR}/components/debug/surgeon

pip install -v ${CURRENT_DIR}/components/profile/benchmark/backend

pip install ${CURRENT_DIR}/components/profile/benchmark