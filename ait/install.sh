#!/usr/bin/env bash

CURRENT_DIR=$(dirname $(readlink -f $0))
arg_force_reinstall=
arg_help=0

while [["$#" -gt 0]]; do case $1 in
  --force-reinstall) arg_force_reinstall=--force-reinstall;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter: $1";
esac; shift; done

pip install CURRENT_DIR arg_force_reinstall

pip install CURRENT_DIR/components/debug/compare arg_force_reinstall

pip install CURRENT_DIR/components/debug arg_force_reinstall

pip install CURRENT_DIR/components/profile/benchmark/backend arg_force_reinstall

pip install CURRENT_DIR/components/profile/benchmark arg_force_reinstall