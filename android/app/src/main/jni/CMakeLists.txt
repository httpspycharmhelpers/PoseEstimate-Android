cmake_minimum_required(VERSION 3.10)
project(yolo11ncnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ncnn and opencv we ship next to this file (prebuilt for arm64-v8a).
set(NCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20260526-android-vulkan/arm64-v8a)
set(OPENCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv-mobile-4.13.0-android/sdk/native)

include_directories(${NCNN_DIR}/include/ncnn)
include_directories(${OPENCV_DIR}/jni/include)

find_library(log_lib log)
find_library(vulkan_lib vulkan)

add_library(ncnn_yolo11 SHARED
        ncnn_bridge.cpp
        yolo11.cpp
        yolo11_det.cpp
        yolo11_seg.cpp
        yolo11_pose.cpp
        yolo11_cls.cpp
        yolo11_obb.cpp)

target_compile_options(ncnn_yolo11 PRIVATE -fopenmp -O3 -fvisibility=hidden -fno-omit-frame-pointer -Wall)

target_link_libraries(ncnn_yolo11
        ${NCNN_DIR}/lib/libncnn.a
        ${OPENCV_DIR}/staticlibs/arm64-v8a/libopencv_core.a
        ${OPENCV_DIR}/staticlibs/arm64-v8a/libopencv_imgproc.a
        ${log_lib}
        ${vulkan_lib}
        -static-libstdc++
        -Wl,--whole-archive
        -Wl,--no-whole-archive)