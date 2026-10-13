# Standalone equivalent of 0510's BUILD_FA_ONLY, using 0520 sources and flags.
# Keep the normal full-kernel build independent of this local compiler override.
cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(sglang_fa3_cu129 LANGUAGES CXX CUDA)

find_package(Python 3.10 COMPONENTS Interpreter Development.SABIModule REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "12.9" OR NOT CUDAToolkit_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "FA3 requires CUDA 12.9, got ${CUDAToolkit_VERSION}")
endif()
if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CUDAToolkit_VERSION)
    message(FATAL_ERROR "FA3 nvcc and CUDA toolkit versions differ")
endif()

# Do not call find_package(Torch): its CUDA-major check rejects this deliberate
# nvcc 12.9 + Torch cu13 combination. Query the selected interpreter instead.
execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/package_fa.py" --describe
    RESULT_VARIABLE TORCH_STATUS OUTPUT_VARIABLE TORCH_INFO OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT TORCH_STATUS EQUAL 0)
    message(FATAL_ERROR "FA3 needs Python >=3.10 with Torch 2.13 and wheel installed")
endif()
string(JSON TORCH_ROOT GET "${TORCH_INFO}" torch_root)
string(JSON TORCH_ABI GET "${TORCH_INFO}" cxx11_abi)
message(STATUS "FA3 host Torch: ${TORCH_INFO}; device nvcc: ${CMAKE_CUDA_COMPILER_VERSION}")

get_filename_component(AOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Match CMake's default in the 0510/0520 full build: static CUDA runtime for
# this extension, avoiding an extra dynamic libcudart.so.12 dependency.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(FA3_CUTLASS_URL "https://github.com/NVIDIA/cutlass/archive/57e3cfb47a2d9e0d46eb6335c3dc411498efa198.tar.gz"
    CACHE STRING "Pinned CUTLASS archive URL (may be a local file)")
set(FA3_ATTENTION_URL "https://github.com/sgl-project/sgl-attn/archive/f89bc2306632d1ec5f97b014dded4254f5b4a907.tar.gz"
    CACHE STRING "Pinned sgl-attn archive URL (may be a local file)")
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
# SOURCE_SUBDIR prevents MakeAvailable from configuring these projects; only
# their source/header files are needed. Standard FETCHCONTENT_SOURCE_DIR_*
# overrides can reuse an already extracted dependency without downloading.
FetchContent_Declare(fa3_cutlass
    URL "${FA3_CUTLASS_URL}"
    URL_HASH SHA256=09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5
    SOURCE_SUBDIR _fa3_source_only
    DOWNLOAD_NO_PROGRESS FALSE TIMEOUT 600 INACTIVITY_TIMEOUT 60
)
FetchContent_Declare(fa3_attention
    URL "${FA3_ATTENTION_URL}"
    URL_HASH SHA256=418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c
    SOURCE_SUBDIR _fa3_source_only
    DOWNLOAD_NO_PROGRESS FALSE TIMEOUT 600 INACTIVITY_TIMEOUT 60
)
message(STATUS "FA3: preparing pinned CUTLASS and sgl-attn sources")
FetchContent_MakeAvailable(fa3_cutlass fa3_attention)

set(FA3_HOPPER "${fa3_attention_SOURCE_DIR}/hopper")
file(GLOB FA3_INSTANTIATIONS
    "${FA3_HOPPER}/instantiations/flash_fwd_hdim[0-9]*_bf16*_sm90.cu"
    "${FA3_HOPPER}/instantiations/flash_fwd_hdimdiff_bf16*_sm90.cu"
    "${FA3_HOPPER}/instantiations/flash_fwd_hdim[0-9]*_fp16*_sm90.cu"
    "${FA3_HOPPER}/instantiations/flash_fwd_hdimdiff_fp16*_sm90.cu"
    "${FA3_HOPPER}/instantiations/flash_fwd_hdim[0-9]*_e4m3*_sm90.cu"
    "${FA3_HOPPER}/instantiations/flash_fwd_hdimdiff_e4m3*_sm90.cu"
)
if(NOT FA3_INSTANTIATIONS)
    message(FATAL_ERROR "No FA3 SM90 instantiations were found")
endif()

# Match the normal 0520 FA3 architecture list. H20-only builds may opt out of
# SM80/86 to reduce build time with -DENABLE_BELOW_SM90=OFF.
set(DEFAULT_ENABLE_BELOW_SM90 ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(DEFAULT_ENABLE_BELOW_SM90 OFF)
endif()
option(ENABLE_BELOW_SM90 "Build SM80/86 kernels as well as SM90a" ${DEFAULT_ENABLE_BELOW_SM90})
set(FA3_GENCODE -gencode=arch=compute_90a,code=sm_90a)
set(FA3_ARCHES "sm_90a")
if(ENABLE_BELOW_SM90)
    file(GLOB FA3_SM8X "${FA3_HOPPER}/instantiations/flash_fwd_hdim*_sm80.cu")
    list(APPEND FA3_INSTANTIATIONS ${FA3_SM8X})
    list(APPEND FA3_GENCODE -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86)
    set(FA3_ARCHES "sm_80,sm_86,sm_90a")
endif()

Python_add_library(flash_ops MODULE USE_SABI 3.10 WITH_SOABI
    "${AOT_SOURCE_DIR}/csrc/flash_extension.cc"
    "${FA3_HOPPER}/flash_api.cpp"
    "${FA3_HOPPER}/flash_prepare_scheduler.cu"
    "${FA3_HOPPER}/flash_fwd_combine.cu"
    ${FA3_INSTANTIATIONS}
)
set_target_properties(flash_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(flash_ops PRIVATE
    "${AOT_SOURCE_DIR}/include"
    "${TORCH_ROOT}/include"
    "${TORCH_ROOT}/include/torch/csrc/api/include"
    ${CUDAToolkit_INCLUDE_DIRS}
    "${fa3_cutlass_SOURCE_DIR}/include"
    "${fa3_cutlass_SOURCE_DIR}/tools/util/include"
    "${FA3_HOPPER}"
)
target_compile_definitions(flash_ops PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI}
    FLASHATTENTION_DISABLE_BACKWARD FLASHATTENTION_DISABLE_DROPOUT
    FLASHATTENTION_DISABLE_UNEVEN_K FLASHATTENTION_VARLEN_ONLY
    FLASHATTENTION_DISABLE_SPARSE_MASK
)
if(NOT ENABLE_BELOW_SM90)
    target_compile_definitions(flash_ops PRIVATE FLASHATTENTION_DISABLE_SM8x)
endif()
set(FA3_CUDA_FLAGS
    -DNDEBUG -DOPERATOR_NAMESPACE=sgl-kernel -O3 -Xcompiler -fPIC
    ${FA3_GENCODE} -std=c++17
    -DCUTE_USE_PACKED_TUPLE=1 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
    -DCUTLASS_VERSIONS_GENERATED -DCUTLASS_TEST_LEVEL=0
    -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0
    -DCUTLASS_ENABLE_GDC_FOR_SM90 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing
)
target_compile_options(flash_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${FA3_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
foreach(name torch torch_cpu torch_cuda c10 c10_cuda)
    # NO_CACHE keeps a reused build directory tied to the selected Python.
    find_library(FA3_${name} NAMES ${name} PATHS "${TORCH_ROOT}/lib" NO_DEFAULT_PATH NO_CACHE REQUIRED)
    target_link_libraries(flash_ops PRIVATE "${FA3_${name}}")
endforeach()
target_link_libraries(flash_ops PRIVATE CUDA::cudart_static CUDA::cuda_driver)

string(JSON BUILD_INFO SET "${TORCH_INFO}" nvcc "\"${CMAKE_CUDA_COMPILER_VERSION}\"")
string(JSON BUILD_INFO SET "${BUILD_INFO}" gpu_arch "\"${FA3_ARCHES}\"")
file(WRITE "${CMAKE_BINARY_DIR}/fa3-build-info.json" "${BUILD_INFO}\n")
list(LENGTH FA3_INSTANTIATIONS FA3_INSTANTIATION_COUNT)
message(STATUS "FA3: ${FA3_INSTANTIATION_COUNT} instantiations for ${FA3_ARCHES}; configuration complete")
