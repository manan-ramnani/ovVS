// Copyright (C) 2021-2025 Intel Corporation
// Copyright (C) 2026 ovVS contributors
// SPDX-License-Identifier: MIT
#pragma once

// Minimal declarations derived from Intel's public level-zero-npu-extensions
// ze_graph_ext.h (https://github.com/intel/level-zero-npu-extensions). The
// ordinary Level Zero SDK does not ship that vendor header. Keep the DDI slots
// in their published order: a wrong slot is an ABI
// violation, not a capability failure.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <level_zero/ze_api.h>

namespace ovvs::npu_escape::l0x {

inline constexpr char kGraphExtensionName[] = "ZE_extension_graph";
inline constexpr std::uint32_t kStructureTypeGraphDesc2 = 0xEu;
inline constexpr std::uint32_t kStructureTypeGraphProperties = 0x3u;
inline constexpr std::uint32_t kStructureTypeGraphProperties2 = 0x10u;
inline constexpr std::uint32_t kStructureTypeGraphArgumentProperties = 0x4u;
inline constexpr std::uint32_t kGraphFormatNative = 0x1u;
inline constexpr std::uint32_t kGraphFlagInputGraphPersistent = 1u << 2;
inline constexpr std::uint32_t kGraphStageCommandListInitialize = 1u << 0;
inline constexpr std::uint32_t kGraphStageInitialize = 1u << 1;
inline constexpr std::uint32_t kGraphArgumentTypeInput = 0u;
inline constexpr std::uint32_t kGraphArgumentTypeOutput = 1u;
inline constexpr std::uint32_t kGraphArgumentPrecisionFp32 = 0x01u;
inline constexpr std::uint32_t kGraphArgumentPrecisionInt32 = 0x05u;
inline constexpr std::size_t kMaxGraphArgumentName = 256;
inline constexpr std::size_t kMaxGraphArgumentDimensions = 5;

struct _ze_graph_handle_t;
using GraphHandle = _ze_graph_handle_t*;
struct _ze_graph_profiling_query_handle_t;
using GraphProfilingQueryHandle = _ze_graph_profiling_query_handle_t*;

struct GraphDesc2 {
  std::uint32_t stype;
  void* pNext;
  std::uint32_t format;
  std::size_t inputSize;
  const std::uint8_t* pInput;
  const char* pBuildFlags;
  std::uint32_t flags;
};

struct GraphProperties {
  std::uint32_t stype;
  void* pNext;
  std::uint32_t numGraphArgs;
};

struct GraphProperties2 {
  std::uint32_t stype;
  void* pNext;
  std::uint32_t numGraphArgs;
  std::uint32_t initStageRequired;
};

struct GraphArgumentProperties {
  std::uint32_t stype;
  void* pNext;
  char name[kMaxGraphArgumentName];
  std::uint32_t type;
  std::uint32_t dims[kMaxGraphArgumentDimensions];
  std::uint32_t networkPrecision;
  std::uint32_t networkLayout;
  std::uint32_t devicePrecision;
  std::uint32_t deviceLayout;
};

using GraphDestroyFn = ze_result_t(ZE_APICALL*)(GraphHandle);
using GraphGetPropertiesFn = ze_result_t(ZE_APICALL*)(GraphHandle, GraphProperties*);
using GraphGetArgumentPropertiesFn =
    ze_result_t(ZE_APICALL*)(GraphHandle, std::uint32_t, GraphArgumentProperties*);
using GraphSetArgumentValueFn =
    ze_result_t(ZE_APICALL*)(GraphHandle, std::uint32_t, const void*);
using AppendGraphInitializeFn = ze_result_t(ZE_APICALL*)(ze_command_list_handle_t, GraphHandle,
                                                         ze_event_handle_t, std::uint32_t,
                                                         ze_event_handle_t*);
using AppendGraphExecuteFn = ze_result_t(ZE_APICALL*)(ze_command_list_handle_t, GraphHandle,
                                                      GraphProfilingQueryHandle, ze_event_handle_t,
                                                      std::uint32_t, ze_event_handle_t*);
using GraphCreate2Fn = ze_result_t(ZE_APICALL*)(ze_context_handle_t, ze_device_handle_t,
                                               const GraphDesc2*, GraphHandle*);
using GraphGetProperties2Fn = ze_result_t(ZE_APICALL*)(GraphHandle, GraphProperties2*);
using GraphInitializeFn = ze_result_t(ZE_APICALL*)(GraphHandle);

struct GraphDdiTable {
  // 1.0
  void* pfnCreate;
  GraphDestroyFn pfnDestroy;
  GraphGetPropertiesFn pfnGetProperties;
  GraphGetArgumentPropertiesFn pfnGetArgumentProperties;
  GraphSetArgumentValueFn pfnSetArgumentValue;
  AppendGraphInitializeFn pfnAppendGraphInitialize;
  AppendGraphExecuteFn pfnAppendGraphExecute;
  void* pfnGetNativeBinary;
  void* pfnDeviceGetGraphProperties;
  // 1.1
  void* pfnGraphGetArgumentMetadata;
  void* pfnGetArgumentProperties2;
  // 1.2
  void* pfnGetArgumentProperties3;
  // 1.3
  void* pfnQueryNetworkCreate;
  void* pfnQueryNetworkDestroy;
  void* pfnQueryNetworkGetSupportedLayers;
  // 1.4
  void* pfnBuildLogGetString;
  // 1.5
  GraphCreate2Fn pfnCreate2;
  void* pfnQueryNetworkCreate2;
  void* pfnQueryContextMemory;
  // 1.6
  void* pfnDeviceGetGraphProperties2;
  // 1.7
  void* pfnGetNativeBinary2;
  // 1.8
  GraphGetProperties2Fn pfnGetProperties2;
  GraphInitializeFn pfnGraphInitialize;
  // 1.11
  void* pfnCompilerGetSupportedOptions;
  void* pfnCompilerIsOptionSupported;
  // 1.12
  void* pfnCreate3;
  void* pfnGetProperties3;
  void* pfnBuildLogGetString2;
  void* pfnBuildLogDestroy;
  // 1.15
  void* pfnSetArgumentValue2;
  // 1.16
  void* pfnEvict;
};

static_assert(std::is_standard_layout_v<GraphDdiTable>);
static_assert(offsetof(GraphDdiTable, pfnCreate2) == 16 * sizeof(void*));
static_assert(offsetof(GraphDdiTable, pfnGetProperties2) == 21 * sizeof(void*));
static_assert(offsetof(GraphDdiTable, pfnGraphInitialize) == 22 * sizeof(void*));

}  // namespace ovvs::npu_escape::l0x
