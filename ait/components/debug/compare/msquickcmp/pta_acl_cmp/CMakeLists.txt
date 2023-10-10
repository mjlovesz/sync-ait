cmake_minimum_required(VERSION 3.0)

project(mylib)

# 添加要编译的源文件
add_library(tensorutil SHARED tensor_util.cpp)

# 设置链接库的路径，这里使用了环境变量
set(ACLTRANSFORMER_HOME_PATH "$ENV{ACLTRANSFORMER_HOME_PATH}")

# 链接所需的库
target_link_libraries(tensorutil PRIVATE "${ACLTRANSFORMER_HOME_PATH}/lib/libacltransformer.so")
target_link_libraries(tensorutil PRIVATE libssl.so)

# 设置输出文件的名称为libtensorutil.so
set_target_properties(tensorutil PROPERTIES OUTPUT_NAME "tensorutil")

# 添加头文件搜索路径
target_include_directories(tensorutil PRIVATE "${ACLTRANSFORMER_HOME_PATH}/../../3rdparty/asdops/include" "${ACLTRANSFORMER_HOME_PATH}/../../core/include")