cmake_minimum_required(VERSION 3.15)
set(LIB_NAME opchecker)

set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_SKIP_RPATH TRUE)

set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
ADD_LINK_OPTIONS(-s)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/dependency
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
    ${CMAKE_CURRENT_LIST_DIR}/atb_util/include
    ${CMAKE_CURRENT_LIST_DIR}/
    )

link_directories(
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
    $ENV{ATB_HOME_PATH}/lib
    )

file(GLOB_RECURSE TEST_UTILS_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/test_utils/context/*.cpp")
file(GLOB_RECURSE ATB_TORCH_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/atb_torch/*.cpp")
file(GLOB_RECURSE DEPEND_SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/dependency/filesystem.cpp")

link_libraries(atb torch_npu)
add_library(${LIB_NAME} SHARED ${TEST_UTILS_SOURCE_FILES} ${ATB_TORCH_SOURCE_FILES} ${ATB_LAYER_OPERATION} ${DEPEND_SOURCE_FILES})

install(
    TARGETS ${LIB_NAME}
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
