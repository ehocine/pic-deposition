cmake_minimum_required(VERSION 3.18)

option(BUILD_CUDA "Build CUDA deposition kernels" OFF)

if(BUILD_CUDA)
    project(pic_deposition LANGUAGES CXX CUDA)
else()
    project(pic_deposition LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    find_library(OPENMP_LIB omp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
    if(OPENMP_LIB)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OPENMP_LIB})
    endif()
endif()

find_package(OpenMP REQUIRED)

find_path(FFTW_INCLUDE_DIR NAMES fftw3.h
    PATHS /usr/include /usr/local/include /opt/homebrew/include)
find_library(FFTW_LIBRARY NAMES fftw3
    PATHS /usr/lib /usr/local/lib /opt/homebrew/lib /usr/lib/x86_64-linux-gnu)
if(NOT FFTW_INCLUDE_DIR OR NOT FFTW_LIBRARY)
    message(FATAL_ERROR "FFTW3 not found. Install with: apt install libfftw3-dev (Linux) or brew install fftw (macOS)")
endif()

set(PIC_SOURCES
    src/pic/field_grid.cpp
    src/pic/particles_aos.cpp
    src/pic/particles_soa.cpp
    src/pic/poisson_fft.cpp
    src/pic/poisson_gs.cpp
    src/pic/diagnostics.cpp
    src/pic/simulation.cpp
    src/pic/validation.cpp
    src/deposition/deposition.cpp
    src/deposition/ngp.cpp
    src/deposition/cic.cpp
    src/deposition/tsc.cpp
    src/deposition/esirkepov.cpp
    src/deposition/deposition_parallel.cpp
    src/sorting/cell_sort.cpp
    src/benchmark/benchmark_runner.cpp
)

add_library(pic_core ${PIC_SOURCES})
target_include_directories(pic_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(pic_core PUBLIC OpenMP::OpenMP_CXX ${FFTW_LIBRARY})
target_include_directories(pic_core PUBLIC ${FFTW_INCLUDE_DIR})

if(BUILD_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        if(DEFINED ENV{CUDAARCHS})
            set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDAARCHS})
        else()
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
    endif()
    target_sources(pic_core PRIVATE
        src/deposition/cuda/deposition_cuda.cu
    )
    target_compile_definitions(pic_core PUBLIC PIC_BUILD_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(pic_core PUBLIC CUDA::cudart)
    set_target_properties(pic_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(pic_sim src/main_sim.cpp)
target_link_libraries(pic_sim PRIVATE pic_core)

add_executable(pic_benchmark src/main_benchmark.cpp)
target_link_libraries(pic_benchmark PRIVATE pic_core)

add_executable(pic_test src/main_test.cpp)
target_link_libraries(pic_test PRIVATE pic_core)

add_executable(pic_validation src/main_validation.cpp)
target_link_libraries(pic_validation PRIVATE pic_core)
