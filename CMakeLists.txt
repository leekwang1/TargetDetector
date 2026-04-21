cmake_minimum_required(VERSION 3.20)

project(DetectionRunner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs objdetect)

if (OpenCV_FOUND)
    set(DETECTION_OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    set(DETECTION_OPENCV_LIBRARIES ${OpenCV_LIBS})
else()
    set(_default_include_dir "")
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/opencv2")
        set(_default_include_dir "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(DETECTION_OPENCV_INCLUDE_DIRS "${_default_include_dir}" CACHE PATH
        "OpenCV include root that contains the opencv2 folder")
    set(DETECTION_OPENCV_LIBRARIES "" CACHE STRING
        "Semicolon-separated OpenCV libraries when OpenCVConfig.cmake is unavailable")

    if (NOT DETECTION_OPENCV_INCLUDE_DIRS OR NOT DETECTION_OPENCV_LIBRARIES)
        message(FATAL_ERROR
            "OpenCV was not found via find_package(). Set OpenCV_DIR to a directory "
            "containing OpenCVConfig.cmake, or set DETECTION_OPENCV_INCLUDE_DIRS and "
            "DETECTION_OPENCV_LIBRARIES manually.")
    endif()
endif()

add_executable(DetectionRunner
    cpp/CenterEstimation.cpp
    cpp/DetectionRunner.cpp
    cpp/PlyWriter.cpp
    cpp/PointCloudRegistration.cpp
)

target_include_directories(DetectionRunner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${DETECTION_OPENCV_INCLUDE_DIRS}
)

if (MSVC)
    target_compile_definitions(DetectionRunner PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

target_link_libraries(DetectionRunner PRIVATE ${DETECTION_OPENCV_LIBRARIES})
