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
  add_test(NAME cpu COMMAND ovvs_tests
    --exclude nndescent_gpu_n_over_4096
    --exclude nndescent_gpu_bounded_scale
    --exclude cagra_sycl_walk_n_over_4096)
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
  add_test(NAME consumer COMMAND ovvs_consumer)
  set_tests_properties(consumer PROPERTIES LABELS "cpu")
  add_test(NAME consumer_cxx COMMAND ovvs_consumer_cxx)
  set_tests_properties(consumer_cxx PROPERTIES LABELS "cpu")
endif()

install(TARGETS ovvs EXPORT ovvsTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/ovvs DESTINATION include)
