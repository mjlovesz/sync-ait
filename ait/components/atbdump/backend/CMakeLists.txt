cmake_minimum_required(VERSION 3.5)
project(MyLibrary)

# 设置编译选项，可以根据需要进行修改
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 添加 use_cxx11_abi 选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# 添加共享库
add_library(MyLibrary SHARED atb_probe.cpp binfile.cpp)

# 设置头文件目录
target_include_directories(MyLibrary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 设置共享库输出目录
set_target_properties(MyLibrary PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# 可选：如果你希望设置.so文件的输出名称，可以使用下面的命令
set_target_properties(MyLibrary PROPERTIES OUTPUT_NAME "atb_probe")