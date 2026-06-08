cmake_minimum_required(VERSION 3.18)
project(pic_deposition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_CUDA "Build CUDA deposition kernels" OFF)

if(APPLE)
    find_library(OPENMP_LIB omp PATHS /opt/homebrew/opt/libomp/lib)
    if(OPENMP_LIB)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OPENMP_LIB})
    endif()
endif()

find_package(OpenMP REQUIRED)

find_path(FFTW_INCLUDE_DIR fftw3.h PATHS /opt/homebrew/include)
find_library(FFTW_LIBRARY fftw3 PATHS /opt/homebrew/lib)
if(NOT FFTW_INCLUDE_DIR OR NOT FFTW_LIBRARY)
    message(FATAL_ERROR "FFTW3 not found. Install with: brew install fftw")
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
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_sources(pic_core PRIVATE
        src/deposition/cuda/deposition_cuda.cu
    )
    target_compile_definitions(pic_core PUBLIC PIC_BUILD_CUDA=1)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(pic_core PUBLIC CUDA::cudart)
endif()

add_executable(pic_sim src/main_sim.cpp)
target_link_libraries(pic_sim PRIVATE pic_core)

add_executable(pic_benchmark src/main_benchmark.cpp)
target_link_libraries(pic_benchmark PRIVATE pic_core)

add_executable(pic_test src/main_test.cpp)
target_link_libraries(pic_test PRIVATE pic_core)

add_executable(pic_validation src/main_validation.cpp)
target_link_libraries(pic_validation PRIVATE pic_core)
