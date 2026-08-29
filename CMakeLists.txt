cmake_minimum_required(VERSION 3.20)
project(ovVS VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if(MSVC)
  foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER ${cfg} cfgU)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfgU} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfgU} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfgU} ${CMAKE_BINARY_DIR}/lib)
  endforeach()
endif()

option(OVVS_WITH_OPENVINO "Build OpenVINO NPU backend" OFF)
option(OVVS_WITH_SYCL "Build SYCL iGPU backend" OFF)
option(OVVS_WITH_MKL "Link oneMKL for GPU GEMM and LAPACK gesvd/syev" OFF)
option(OVVS_BUILD_TESTS "Build unit tests" ON)
option(OVVS_BUILD_TOOLS "Build probe/bakeoff tools" ON)
option(OVVS_BUILD_ESCAPE_EXPERIMENTS
  "Enable eligible retained NPU/iGPU experiments (targets still require their SYCL/OpenVINO/platform dependencies)"
  OFF)

if(MSVC)
  add_compile_options(/W3 /EHsc /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(OVVS_SOURCES
  src/rt/status.cpp
  src/rt/probe.cpp
  src/rt/resources.cpp
  compiler/shave/topk.c
  compiler/shave/adc.c
  src/prim/cpu/cpu_prims.cpp
  src/prim/mixer.cpp
  src/prim/npu/backend_npu.cpp
  src/prim/gpu/backend_gpu.cpp
  src/neighbors/indexes.cpp
  src/neighbors/graphs.cpp
  src/cluster/cluster.cpp
  src/preprocess/preprocess.cpp
  src/rt/batcher.cpp
  src/rt/energy.cpp
  src/rt/shave_elf.cpp
)

add_library(ovvs SHARED ${OVVS_SOURCES})
target_include_directories(ovvs
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler/shave
)
target_compile_definitions(ovvs PRIVATE OVVS_BUILD)

if(WIN32)
  target_link_libraries(ovvs PRIVATE advapi32 setupapi pdh)
endif()

if(OVVS_WITH_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  target_link_libraries(ovvs PRIVATE openvino::runtime)
  target_compile_definitions(ovvs PRIVATE OVVS_WITH_OPENVINO)
  if(WIN32)
    add_custom_command(TARGET ovvs POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:openvino::runtime>
        $<TARGET_FILE_DIR:ovvs>
      COMMENT "Copy OpenVINO runtime next to ovvs")
  endif()
endif()

if(OVVS_WITH_SYCL)
  # Windows oneAPI: configure with icx (MSVC-like). icpx is GNU-like and fails CMake MSVC flags.
  target_compile_definitions(ovvs PRIVATE OVVS_WITH_SYCL)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    target_compile_options(ovvs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
    target_link_options(ovvs PRIVATE -fsycl)
  endif()
endif()

if(OVVS_WITH_MKL)
  find_package(MKL CONFIG REQUIRED)
  target_compile_definitions(ovvs PRIVATE OVVS_WITH_MKL)
  if(TARGET MKL::MKL_SYCL)
    target_link_libraries(ovvs PRIVATE MKL::MKL_SYCL)
  endif()
  if(TARGET MKL::MKL)
    target_link_libraries(ovvs PRIVATE MKL::MKL)
  endif()
endif()

add_library(ovvs::ovvs ALIAS ovvs)

if(OVVS_BUILD_TOOLS)
  add_executable(ovvs_probe tools/probe/main.cpp)
  target_link_libraries(ovvs_probe PRIVATE ovvs)
  add_executable(ovvs_bakeoff tools/bakeoff/main.cpp)
  target_link_libraries(ovvs_bakeoff PRIVATE ovvs)
endif()

if(OVVS_BUILD_ESCAPE_EXPERIMENTS)
  if(OVVS_WITH_SYCL AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    add_executable(ovvs_gpu_root_group_canary
      tools/gpu_root_group_canary/main.cpp)
    target_compile_options(ovvs_gpu_root_group_canary PRIVATE -fsycl)
    target_link_options(ovvs_gpu_root_group_canary PRIVATE -fsycl)

    add_executable(ovvs_pq_gpu_fused_bench tools/pq_gpu_fused_bench/main.cpp)
    target_compile_options(ovvs_pq_gpu_fused_bench PRIVATE -fsycl)
    target_link_options(ovvs_pq_gpu_fused_bench PRIVATE -fsycl)

    if(OVVS_WITH_OPENVINO)
      add_executable(ovvs_hetero_ivfpq_bench tools/hetero_ivfpq_bench/main.cpp)
      target_compile_options(ovvs_hetero_ivfpq_bench PRIVATE -fsycl)
      target_link_options(ovvs_hetero_ivfpq_bench PRIVATE -fsycl)
      target_link_libraries(ovvs_hetero_ivfpq_bench PRIVATE ovvs)
    endif()
  endif()

  if(WIN32 AND OVVS_WITH_OPENVINO AND OVVS_WITH_SYCL AND
     CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    add_executable(ovvs_pq_lut_factor_bench tools/pq_lut_factor_bench/main.cpp)
    target_compile_options(ovvs_pq_lut_factor_bench PRIVATE -fsycl)
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      target_compile_options(ovvs_pq_lut_factor_bench PRIVATE
        /fp:precise
        /clang:-fno-fast-math
        /clang:-fno-associative-math
        /clang:-ffp-contract=off)
    else()
      target_compile_options(ovvs_pq_lut_factor_bench PRIVATE
        -ffp-model=precise
        -fno-fast-math
        -fno-associative-math
        -ffp-contract=off)
    endif()
    target_link_options(ovvs_pq_lut_factor_bench PRIVATE -fsycl)
    target_link_libraries(ovvs_pq_lut_factor_bench PRIVATE openvino::runtime psapi)
  endif()

  if(WIN32 AND OVVS_WITH_OPENVINO)
    find_path(OVVS_LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h
      HINTS "$ENV{LEVEL_ZERO_V1_SDK_PATH}/include"
      PATHS "C:/Program Files/LevelZeroSDK/1.32.0/include")
    if(OVVS_LEVEL_ZERO_INCLUDE_DIR)
      add_executable(ovvs_npu_l0_escape tools/npu_escape/main.cpp)
      target_include_directories(ovvs_npu_l0_escape PRIVATE
        ${OVVS_LEVEL_ZERO_INCLUDE_DIR})
      target_link_libraries(ovvs_npu_l0_escape PRIVATE openvino::runtime psapi)
      add_custom_command(TARGET ovvs_npu_l0_escape POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:openvino::runtime>
          $<TARGET_FILE_DIR:ovvs_npu_l0_escape>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE_DIR:openvino::runtime>/openvino_intel_npu_plugin.dll
          $<TARGET_FILE_DIR:ovvs_npu_l0_escape>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE_DIR:openvino::runtime>/tbb12.dll
          $<TARGET_FILE_DIR:ovvs_npu_l0_escape>
        COMMENT "Copy OpenVINO runtime next to the direct Level Zero experiment")
    else()
      message(STATUS "Level Zero SDK headers not found; omitting ovvs_npu_l0_escape")
    endif()
  endif()
endif()

if(OVVS_BUILD_TESTS)
  add_executable(ovvs_tests
    tests/test_main.cpp
    tests/test_prim.cpp
    tests/test_neighbors.cpp
    tests/test_cluster.cpp
  )
  target_link_libraries(ovvs_tests PRIVATE ovvs)
  if(WIN32)
    target_link_options(ovvs_tests PRIVATE /STACK:16777216)
  endif()
  target_include_directories(ovvs_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_executable(ovvs_consumer tests/consumer.c)
  target_link_libraries(ovvs_consumer PRIVATE ovvs)
  add_executable(ovvs_consumer_cxx tests/consumer.cpp)
  target_link_libraries(ovvs_consumer_cxx PRIVATE ovvs)
  enable_testing()
  if(TARGET ovvs_gpu_root_group_canary)
    add_test(NAME gpu_root_group_canary
      COMMAND ovvs_gpu_root_group_canary)
    set_tests_properties(gpu_root_group_canary PROPERTIES
      LABELS "gpu;experimental;capability"
      RESOURCE_LOCK intel_gpu
      SKIP_RETURN_CODE 77
      TIMEOUT 20)
  endif()
  add_test(NAME cpu COMMAND ovvs_tests
    --exclude nndescent_gpu_n_over_4096
    --exclude nndescent_gpu_bounded_scale
    --exclude cagra_sycl_walk_n_over_4096
    --exclude ivf_pq_fused_gpu_parity_filters_tail_and_blocks
    --exclude ivf_pq_fused_gpu_krefine_ties_and_atomicity
    --exclude ivf_pq_fused_gpu_cross_list_equal_score_merge
    --exclude ivf_pq_fused_gpu_persistence_extend_and_concurrency)
  set_tests_properties(cpu PROPERTIES
    LABELS "cpu"
    RESOURCE_LOCK intel_gpu)
  add_test(NAME nndescent_gpu_n_over_4096
    COMMAND ovvs_tests --filter nndescent_gpu_n_over_4096)
  set_tests_properties(nndescent_gpu_n_over_4096 PROPERTIES
    LABELS "gpu;scale"
    RESOURCE_LOCK intel_gpu
    SKIP_RETURN_CODE 77
    TIMEOUT 45)
  add_test(NAME nndescent_gpu_bounded_scale
    COMMAND ovvs_tests --filter nndescent_gpu_bounded_scale)
  set_tests_properties(nndescent_gpu_bounded_scale PROPERTIES
    LABELS "gpu;scale"
    RESOURCE_LOCK intel_gpu
    SKIP_RETURN_CODE 77
    TIMEOUT 120)
  add_test(NAME cagra_sycl_walk_n_over_4096
    COMMAND ovvs_tests --filter cagra_sycl_walk_n_over_4096)
  set_tests_properties(cagra_sycl_walk_n_over_4096 PROPERTIES
    LABELS "gpu;scale"
    RESOURCE_LOCK intel_gpu
    SKIP_RETURN_CODE 77
    TIMEOUT 45)
  add_test(NAME ivf_pq_fused_gpu_scan_select
    COMMAND ovvs_tests
      --filter ivf_pq_fused_gpu_parity_filters_tail_and_blocks
      --filter ivf_pq_fused_gpu_krefine_ties_and_atomicity
      --filter ivf_pq_fused_gpu_cross_list_equal_score_merge
      --filter ivf_pq_fused_gpu_persistence_extend_and_concurrency)
  set_tests_properties(ivf_pq_fused_gpu_scan_select PROPERTIES
    LABELS "gpu;ivf-pq"
    RESOURCE_LOCK intel_gpu
    SKIP_RETURN_CODE 77
    TIMEOUT 120)
  add_test(NAME consumer COMMAND ovvs_consumer)
  set_tests_properties(consumer PROPERTIES LABELS "cpu")
  add_test(NAME consumer_cxx COMMAND ovvs_consumer_cxx)
  set_tests_properties(consumer_cxx PROPERTIES LABELS "cpu")
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME hnswlib_interop
      COMMAND ${CMAKE_COMMAND} -E env
        "OVVS_LIBRARY=$<TARGET_FILE:ovvs>"
        "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/hnswlib_interop.py)
    set_tests_properties(hnswlib_interop PROPERTIES
      LABELS "cpu;interop"
      SKIP_RETURN_CODE 77
      TIMEOUT 120)
  endif()
endif()

install(TARGETS ovvs EXPORT ovvsTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/ovvs DESTINATION include)
