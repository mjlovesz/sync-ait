#!/bin/bash

# 进入ss目录
cd pta_acl_cmp

# 设置CMake构建目录
build_dir="build"

# 检查构建目录是否存在，如果不存在则创建
if [ ! -d "$build_dir" ]; then
    mkdir "$build_dir"
fi

# 进入CMake构建目录
cd "$build_dir"

# 调用CMake来构建项目
cmake ..

# 使用make来编译项目
make

site_packages_path=$(python -c "import site; print(site.getsitepackages()[0])")
# 指定.so文件的目标目录

# 检查目标目录是否存在，如果不存在则创建
if [ ! -d "$site_packages_path" ]; then
    mkdir -p "$site_packages_path"
fi

# 将生成的.so文件移动到目标目录
mv libtensorutil.so "${site_packages_path}/msquickcmp"


# 返回原始目录
cd ../..

# 完成
echo "Build and move completed!"