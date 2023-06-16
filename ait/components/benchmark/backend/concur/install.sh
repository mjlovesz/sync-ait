#!/bin/bash

if [ -d "build" ]; then
  echo "Directory 'build' already exists."
else
  mkdir build
  echo "Directory 'build' created."
fi

cd build
cmake ..
make
pwd=$(pwd)
export PATH="/$pwd:$PATH"