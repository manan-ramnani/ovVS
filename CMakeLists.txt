cmake_minimum_required(VERSION 3.20)
project(ioVS VERSION 0.2.0 LANGUAGES C CXX)

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

option(IOVS_WITH_OPENVINO "Build OpenVINO NPU backend" OFF)
option(IOVS_WITH_SYCL "Build SYCL iGPU backend" OFF)
option(IOVS_BUILD_TESTS "Build unit tests" ON)
option(IOVS_BUILD_TOOLS "Build probe/bakeoff tools" ON)

if(MSVC)
  add_compile_options(/W3 /EHsc /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

set(IOVS_SOURCES
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
)

add_library(iovs SHARED ${IOVS_SOURCES})
target_include_directories(iovs
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler/shave
)
target_compile_definitions(iovs PRIVATE IOVS_BUILD)

if(WIN32)
  target_link_libraries(iovs PRIVATE advapi32)
endif()

if(IOVS_WITH_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  target_link_libraries(iovs PRIVATE openvino::runtime)
  target_compile_definitions(iovs PRIVATE IOVS_WITH_OPENVINO)
endif()

if(IOVS_WITH_SYCL)
  target_compile_definitions(iovs PRIVATE IOVS_WITH_SYCL)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    target_compile_options(iovs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsycl>)
    target_link_options(iovs PRIVATE -fsycl)
  endif()
endif()

add_library(iovs::iovs ALIAS iovs)

if(IOVS_BUILD_TOOLS)
  add_executable(iovs_probe tools/probe/main.cpp)
  target_link_libraries(iovs_probe PRIVATE iovs)
  add_executable(iovs_bakeoff tools/bakeoff/main.cpp)
  target_link_libraries(iovs_bakeoff PRIVATE iovs)
endif()

if(IOVS_BUILD_TESTS)
  add_executable(iovs_tests
    tests/test_main.cpp
    tests/test_prim.cpp
    tests/test_neighbors.cpp
    tests/test_cluster.cpp
  )
  target_link_libraries(iovs_tests PRIVATE iovs)
  if(MSVC)
    target_link_options(iovs_tests PRIVATE /STACK:16777216)
  endif()
  target_include_directories(iovs_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_executable(iovs_consumer tests/consumer.c)
  target_link_libraries(iovs_consumer PRIVATE iovs)
  add_executable(iovs_consumer_cxx tests/consumer.cpp)
  target_link_libraries(iovs_consumer_cxx PRIVATE iovs)
  enable_testing()
  add_test(NAME cpu COMMAND iovs_tests)
  set_tests_properties(cpu PROPERTIES LABELS "cpu")
  add_test(NAME consumer COMMAND iovs_consumer)
  set_tests_properties(consumer PROPERTIES LABELS "cpu")
  add_test(NAME consumer_cxx COMMAND iovs_consumer_cxx)
  set_tests_properties(consumer_cxx PROPERTIES LABELS "cpu")
endif()

install(TARGETS iovs EXPORT iovsTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/iovs DESTINATION include)
