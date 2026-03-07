cmake_minimum_required(VERSION 3.15...4.2)

project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.7 COMPONENTS Development.Module REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v3.0.2
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# Auto-detect CUDA if not explicitly set via env var
if(DEFINED ENV{DIFFVG_CUDA})
    if("$ENV{DIFFVG_CUDA}" STREQUAL "1" OR "$ENV{DIFFVG_CUDA}" STREQUAL "ON")
        set(DIFFVG_CUDA ON CACHE BOOL "Build diffvg with GPU code path?")
    else()
        set(DIFFVG_CUDA OFF CACHE BOOL "Build diffvg with GPU code path?")
    endif()
else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(DIFFVG_CUDA ON CACHE BOOL "Build diffvg with GPU code path?")
    else()
        set(DIFFVG_CUDA OFF CACHE BOOL "Build diffvg with GPU code path?")
    endif()
endif()

if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
else()
    FetchContent_Declare(
      thrust
      GIT_REPOSITORY https://github.com/NVIDIA/cccl
      GIT_TAG        v3.3.0
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(thrust)
endif()

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${PYTHON_INCLUDE_DIRS})

if(NOT MSVC)
  # These compile definitions are not meaningful for MSVC
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
endif()

if(NOT DIFFVG_CUDA)
    add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS atomic.h
         color.h
         cdf.h
         cuda_utils.h
         diffvg.h
         edge_query.h
         filter.h
         matrix.h
         parallel.h
         pcg.h
         ptr.h
         sample_boundary.h
         scene.h
         shape.h
         solve.h
         vector.h
         within_distance.h
         winding_number.h
         atomic.cpp
         color.cpp
         diffvg.cpp
         parallel.cpp
         scene.cpp
         shape.cpp)

if(DIFFVG_CUDA)
    set_source_files_properties(diffvg.cpp scene.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(diffvg MODULE ${SRCS})

target_link_libraries(diffvg PUBLIC pybind11::module)

if(DIFFVG_CUDA)
    link_directories(${CUDA_LIBRARIES})
else()
    target_include_directories(diffvg PRIVATE "${thrust_SOURCE_DIR}/thrust")
    target_include_directories(diffvg PRIVATE "${thrust_SOURCE_DIR}/libcudacxx/include")
endif()

if(APPLE)
    # The "-undefined dynamic_lookup" is a hack for systems with
    # multiple Python installed. If we link a particular Python version
    # here, and we import it with a different Python version later.
    # likely a segmentation fault.
    # The solution for Linux Mac OS machines, as mentioned in 
    # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
    # is to not link against Python library at all and resolve the symbols
    # at compile time.
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()

target_link_options(diffvg PUBLIC ${DYNAMIC_LOOKUP})

if(WIN32)
    # See: https://pybind11.readthedocs.io/en/master/compiling.html#advanced-interface-library-target
    set_target_properties(diffvg PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

set_target_properties(diffvg PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(diffvg PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(UNIX AND NOT APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "@loader_path")
endif()

set_property(TARGET diffvg PROPERTY CXX_STANDARD 17)
set_target_properties(diffvg PROPERTIES PREFIX "")
# Still enable assertion in release mode
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

install(TARGETS diffvg LIBRARY DESTINATION .)
