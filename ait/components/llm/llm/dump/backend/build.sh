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


CUR_PATH=$(dirname $(readlink -f $0))

# 设置CMake构建目录
build_dir="${CUR_PATH}/build"

# 检查构建目录是否存在，如果不存在则创建
if [ ! -d "$build_dir" ]; then
    mkdir "$build_dir"
    chmod 750 $build_dir
fi

# 进入CMake构建目录
cd "$build_dir"

# 调用CMake来构建项目
cmake ..

# 使用make来编译项目
make

site_packages_path=$(python3 -c "import site; print(site.getsitepackages()[0])")
# 指定.so文件的目标目录

# 检查目标目录是否存在，如果不存在则创建
if [ ! -d "${site_packages_path}/llm/dump/backend/lib" ]; then
    mkdir -p "${site_packages_path}/llm/dump/backend/lib"
    chmod 750 ${site_packages_path}/llm/dump/backend/lib
fi

# 将生成的.so文件移动到目标目录
mv ../lib/libatb_probe.so "${site_packages_path}/llm/dump/backend/lib"


# 返回原始目录
cd ..

# 完成
echo "Build and move to ${site_packages_path}/llm/dump/backend/lib completed!"