cmake_minimum_required(VERSION 3.0)

project(mylib)

# 添加要编译的源文件
add_library(savetensor SHARED save_tensor.cpp)

# 设置链接库的路径，这里使用了环境变量
set(ATB_HOME_PATH "$ENV{ATB_HOME_PATH}")

# 链接所需的库
target_link_libraries(savetensor PRIVATE "${ATB_HOME_PATH}/lib/libatb_static.a")
target_link_libraries(savetensor PRIVATE "${ATB_HOME_PATH}/lib/libasdops_static.a")
target_link_libraries(savetensor PRIVATE libssl.so)

# 设置输出文件的名称为libsavetensor.so
set_target_properties(savetensor PROPERTIES OUTPUT_NAME "savetensor")
