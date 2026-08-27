#include "internal.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <mutex>

namespace {

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    if (c == '\\' || c == '"') {
      o.push_back('\\');
      o.push_back(static_cast<char>(c));
    } else if (c >= 0x20) {
      o.push_back(static_cast<char>(c));
    }
  }
  return o;
}

std::string vcl_result_name(uint32_t rc) {
  switch (rc) {
    case 0:
      return "success";
    case 0x70000002u:
      return "out_of_memory";
    case 0x78000003u:
      return "unsupported_feature";
    case 0x78000004u:
      return "invalid_argument";
    case 0x78000005u:
      return "invalid_null_handle";
    case 0x78000006u:
      return "io";
    case 0x78000007u:
      return "invalid_ir";
    case 0x7ffffffeu:
      return "unknown";
    case 0xffffffffu:
      return "exception";
    default: {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "0x%08x", rc);
      return buf;
    }
  }
}

#ifdef _WIN32
struct vcl_version_info_t {
  uint16_t major;
  uint16_t minor;
};
enum vcl_log_level_t { kVclLogNone = 0, kVclLogError = 1 };
struct vcl_compiler_desc_t {
  vcl_version_info_t version;
  vcl_log_level_t debugLevel;
};
struct vcl_device_desc_t {
  uint64_t size;
  uint32_t deviceID;
  uint16_t revision;
  uint32_t tileCount;
};
struct vcl_executable_desc_t {
  const uint8_t* modelIRData;
  uint64_t modelIRSize;
  const char* options;
  uint64_t optionsSize;
};

using vcl_get_version_fn = uint32_t(__cdecl*)(vcl_version_info_t*, vcl_version_info_t*);
using vcl_compiler_create_fn = uint32_t(__cdecl*)(vcl_compiler_desc_t*, vcl_device_desc_t*, void**, void**);
using vcl_compiler_destroy_fn = uint32_t(__cdecl*)(void*);
using vcl_exec_create_fn = uint32_t(__cdecl*)(void*, vcl_executable_desc_t, void**);
using vcl_exec_destroy_fn = uint32_t(__cdecl*)(void*);

uint32_t vcl_create_seh(vcl_compiler_create_fn fn, vcl_compiler_desc_t* desc, vcl_device_desc_t* dev, void** compiler,
                        void** log) {
  __try {
    return fn(desc, dev, compiler, log);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return 0xffffffffu;
  }
}

uint32_t vcl_exec_create_seh(vcl_exec_create_fn fn, void* compiler, vcl_executable_desc_t desc, void** exe) {
  __try {
    return fn(compiler, desc, exe);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return 0xffffffffu;
  }
}

std::string find_npu_compiler_dll() {
  WIN32_FIND_DATAA fd{};
  HANDLE h = FindFirstFileA("C:\\Windows\\System32\\DriverStore\\FileRepository\\npu.inf_*", &fd);
  if (h == INVALID_HANDLE_VALUE) return {};
  std::string loader;
  std::string vcl;
  do {
    if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) continue;
    const std::string dir =
        std::string("C:\\Windows\\System32\\DriverStore\\FileRepository\\") + fd.cFileName;
    const std::string a = dir + "\\openvino_intel_npu_compiler_loader.dll";
    const std::string b = dir + "\\npu_driver_compiler.dll";
    if (GetFileAttributesA(a.c_str()) != INVALID_FILE_ATTRIBUTES) loader = a;
    if (GetFileAttributesA(b.c_str()) != INVALID_FILE_ATTRIBUTES) vcl = b;
  } while (FindNextFileA(h, &fd));
  FindClose(h);
  if (!loader.empty()) return loader;
  return vcl;
}

std::string firmware_names(const std::string& compiler_dll) {
  const auto slash = compiler_dll.find_last_of("\\/");
  if (slash == std::string::npos) return "[]";
  const std::string dir = compiler_dll.substr(0, slash) + "\\firmware\\*.bin";
  WIN32_FIND_DATAA fd{};
  HANDLE h = FindFirstFileA(dir.c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) return "[]";
  std::string o = "[";
  bool first = true;
  do {
    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
    if (!first) o += ", ";
    first = false;
    o += "\"";
    o += json_escape(fd.cFileName);
    o += "\"";
  } while (FindNextFileA(h, &fd));
  FindClose(h);
  o += "]";
  return o;
}

std::vector<std::string> kernel_search_roots() {
  char mod[MAX_PATH] = {0};
  std::vector<std::string> roots;
  if (GetModuleFileNameA(nullptr, mod, MAX_PATH)) {
    std::string dir(mod);
    const auto slash = dir.find_last_of("\\/");
    if (slash != std::string::npos) dir.resize(slash);
    roots.push_back(dir);
    roots.push_back(dir + "\\..");
    roots.push_back(dir + "\\..\\..");
    roots.push_back(dir + "\\..\\..\\..");
  }
  roots.emplace_back(".");
  return roots;
}

std::string find_kernel_elf(const char* name) {
  const std::string rel =
      std::string("compiler\\npu_compiler\\sw_runtime_kernels\\kernels\\prebuild\\act_shave_bin\\") + name;
  for (const auto& r : kernel_search_roots()) {
    const std::string p = r + "\\" + rel;
    if (GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES) return p;
  }
  return {};
}

std::string find_signed_elf() {
  const char* names[] = {"gather.3720xx.elf", "activation_abs.3720xx.elf", "activation_abs.4000xx.elf"};
  for (const char* n : names) {
    std::string p = find_kernel_elf(n);
    if (!p.empty()) return p;
  }
  return {};
}

std::string ze_result_name(uint32_t rc) {
  switch (rc) {
    case 0:
      return "success";
    case 0x78000003u:
      return "unsupported_feature";
    case 0x78000004u:
      return "invalid_argument";
    case 0x78000007u:
      return "invalid_null_pointer";
    case 0x7800000fu:
      return "invalid_native_binary";
    case 0x78000001u:
      return "uninitialized";
    case 0x7ffffffeu:
      return "unknown";
    default: {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "0x%08x", rc);
      return buf;
    }
  }
}

struct ze_graph_desc_2_t {
  uint32_t stype;
  void* pNext;
  uint32_t format;
  size_t inputSize;
  const uint8_t* pInput;
  const char* pBuildFlags;
  uint32_t flags;
};

struct ze_activation_kernel_desc_t {
  uint32_t stype;
  void* pNext;
  size_t kernelDataSize;
  const uint8_t* pKernelData;
};

struct ze_graph_ddi_t {
  uint32_t(__cdecl* pfnCreate)(void*, void*, const void*, void**);
  uint32_t(__cdecl* pfnDestroy)(void*);
  void* unused[7];
  void* v11[2];
  void* v12[1];
  void* v13[3];
  void* v14[1];
  uint32_t(__cdecl* pfnCreate2)(void*, void*, const ze_graph_desc_2_t*, void**);
};

struct L0GraphHits {
  std::string ext;
  std::string native_blob = "not_attempted";
  std::string native_shave_elf = "not_attempted";
  std::string native_unsigned_elf = "not_attempted";
  std::string actkernel_unsigned = "not_attempted";
};

uint32_t l0_graph_create2_seh(ze_graph_ddi_t* ddi, void* ctx, void* dev, const ze_graph_desc_2_t* desc, void** graph) {
  __try {
    if (!ddi || !ddi->pfnCreate2) return 0x78000003u;
    return ddi->pfnCreate2(ctx, dev, desc, graph);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return 0xffffffffu;
  }
}

L0GraphHits l0_try_graphs(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& shave_elf,
                          const std::vector<uint8_t>& unsigned_elf) {
  L0GraphHits out;
  HMODULE ze = LoadLibraryA("ze_loader.dll");
  if (!ze) {
    out.ext = "no_ze_loader";
    return out;
  }
  using zeInit_fn = uint32_t(__cdecl*)(uint32_t);
  using zeDriverGet_fn = uint32_t(__cdecl*)(uint32_t*, void**);
  using zeDeviceGet_fn = uint32_t(__cdecl*)(void*, uint32_t*, void**);
  using zeContextCreate_fn = uint32_t(__cdecl*)(void*, const void*, void**);
  using zeContextDestroy_fn = uint32_t(__cdecl*)(void*);
  using zeGetExt_fn = uint32_t(__cdecl*)(void*, const char*, void**);
  auto zinit = reinterpret_cast<zeInit_fn>(GetProcAddress(ze, "zeInit"));
  auto zdrivers = reinterpret_cast<zeDriverGet_fn>(GetProcAddress(ze, "zeDriverGet"));
  auto zdevices = reinterpret_cast<zeDeviceGet_fn>(GetProcAddress(ze, "zeDeviceGet"));
  auto zctx = reinterpret_cast<zeContextCreate_fn>(GetProcAddress(ze, "zeContextCreate"));
  auto zctxd = reinterpret_cast<zeContextDestroy_fn>(GetProcAddress(ze, "zeContextDestroy"));
  auto zext = reinterpret_cast<zeGetExt_fn>(GetProcAddress(ze, "zeDriverGetExtensionFunctionAddress"));
  if (!zinit || !zdrivers || !zdevices || !zctx || !zext || zinit(0) != 0) {
    out.ext = "ze_init_failed";
    FreeLibrary(ze);
    return out;
  }
  uint32_t ndrv = 0;
  zdrivers(&ndrv, nullptr);
  if (ndrv == 0 || ndrv > 16) {
    out.ext = "no_drivers";
    FreeLibrary(ze);
    return out;
  }
  std::vector<void*> drvs(ndrv);
  zdrivers(&ndrv, drvs.data());
  ze_graph_ddi_t* ddi = nullptr;
  void* driver = nullptr;
  void* device = nullptr;
  void* ctx = nullptr;
  for (uint32_t i = 0; i < ndrv && !ddi; ++i) {
    void* table = nullptr;
    if (zext(drvs[i], "ZE_extension_graph", &table) != 0 || !table) continue;
    ddi = reinterpret_cast<ze_graph_ddi_t*>(table);
    driver = drvs[i];
    uint32_t ndev = 0;
    zdevices(driver, &ndev, nullptr);
    if (ndev == 0 || ndev > 32) continue;
    std::vector<void*> devs(ndev);
    zdevices(driver, &ndev, devs.data());
    device = devs[0];
  }
  if (!ddi || !driver || !device) {
    out.ext = "no_graph_ext";
    FreeLibrary(ze);
    return out;
  }
  out.ext = "ZE_extension_graph";
  struct {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
  } cdesc{0xd, nullptr, 0};
  if (zctx(driver, &cdesc, &ctx) != 0 || !ctx) {
    out.ext += "+ctx_fail";
    FreeLibrary(ze);
    return out;
  }
  auto try_native = [&](const std::vector<uint8_t>& bytes, void* pnext) -> std::string {
    if (bytes.empty()) return "empty";
    ze_graph_desc_2_t desc{};
    desc.stype = 0xE;
    desc.pNext = pnext;
    desc.format = 0x1;
    desc.inputSize = bytes.size();
    desc.pInput = bytes.data();
    desc.pBuildFlags = nullptr;
    desc.flags = 0;
    void* graph = nullptr;
    const uint32_t rc = l0_graph_create2_seh(ddi, ctx, device, &desc, &graph);
    if (graph && ddi->pfnDestroy) ddi->pfnDestroy(graph);
    return ze_result_name(rc);
  };
  out.native_blob = try_native(blob, nullptr);
  out.native_shave_elf = try_native(shave_elf, nullptr);
  out.native_unsigned_elf = try_native(unsigned_elf, nullptr);
  if (!unsigned_elf.empty()) {
    ze_activation_kernel_desc_t act{};
    act.stype = 0x5;
    act.pNext = nullptr;
    act.kernelDataSize = unsigned_elf.size();
    act.pKernelData = unsigned_elf.data();
    ze_graph_desc_2_t desc{};
    desc.stype = 0xE;
    desc.pNext = &act;
    desc.format = 0x2; /* NGRAPH_LITE — activation kernel is a compile-time extra, not a native-blob field */
    desc.inputSize = unsigned_elf.size();
    desc.pInput = unsigned_elf.data();
    desc.pBuildFlags = nullptr;
    desc.flags = 0;
    void* graph = nullptr;
    const uint32_t rc = l0_graph_create2_seh(ddi, ctx, device, &desc, &graph);
    if (graph && ddi->pfnDestroy) ddi->pfnDestroy(graph);
    out.actkernel_unsigned = ze_result_name(rc);
  }
  if (zctxd && ctx) zctxd(ctx);
  FreeLibrary(ze);
  return out;
}

bool has_movicompile() {
  char buf[MAX_PATH];
  return SearchPathA(nullptr, "moviCompile.exe", nullptr, MAX_PATH, buf, nullptr) != 0 ||
         SearchPathA(nullptr, "moviCompile", nullptr, MAX_PATH, buf, nullptr) != 0;
}

std::string silicon_evidence_fields() {
  int shave_n = 0, dpu_n = 0;
  std::vector<uint8_t> blob;
  std::string exec;
  const bool ran = ovvs::impl::npu_shave_profile_adc(&shave_n, &dpu_n, &blob, &exec);
  std::vector<uint8_t> signed_bytes;
  const std::string signed_path = find_signed_elf();
  if (!signed_path.empty()) {
    std::ifstream f(signed_path, std::ios::binary);
    signed_bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  uint8_t raw[64] = {0};
  raw[0] = 0x7f;
  raw[1] = 'E';
  raw[2] = 'L';
  raw[3] = 'F';
  raw[4] = 1;
  raw[5] = 1;
  raw[6] = 1;
  raw[18] = 2;
  std::vector<uint8_t> unsigned_elf(raw, raw + sizeof(raw));
  const L0GraphHits l0 = l0_try_graphs(blob, signed_bytes, unsigned_elf);
  const bool has_kernel_text = std::search(blob.begin(), blob.end(),
                                           reinterpret_cast<const uint8_t*>(".text.KernelText"),
                                           reinterpret_cast<const uint8_t*>(".text.KernelText") + 16) != blob.end();
  const bool has_act_inv =
      std::search(blob.begin(), blob.end(), reinterpret_cast<const uint8_t*>(".text.ActKernelInvocations"),
                  reinterpret_cast<const uint8_t*>(".text.ActKernelInvocations") + 26) != blob.end();
  std::string silicon = "unsupported_no_inject_api";
  if (ran && shave_n > 0) silicon = "compiler_actshave";
  std::ostringstream o;
  o << "  \"shave_tasks\": " << shave_n << ",\n";
  o << "  \"shave_dpu_tasks\": " << dpu_n << ",\n";
  o << "  \"shave_exec_types\": \"" << json_escape(exec) << "\",\n";
  o << "  \"shave_graph_blob_bytes\": " << blob.size() << ",\n";
  o << "  \"shave_graph_has_kernel_text\": " << (has_kernel_text ? "true" : "false") << ",\n";
  o << "  \"shave_graph_has_act_invocations\": " << (has_act_inv ? "true" : "false") << ",\n";
  o << "  \"shave_l0_ext\": \"" << json_escape(l0.ext) << "\",\n";
  o << "  \"shave_l0_native_blob\": \"" << json_escape(l0.native_blob) << "\",\n";
  o << "  \"shave_l0_native_elf32\": \"" << json_escape(l0.native_shave_elf) << "\",\n";
  o << "  \"shave_l0_unsigned_elf32\": \"" << json_escape(l0.native_unsigned_elf) << "\",\n";
  o << "  \"shave_l0_actkernel_unsigned\": \"" << json_escape(l0.actkernel_unsigned) << "\",\n";
  o << "  \"shave_unsigned_inject\": \"unsupported_no_inject_api\",\n";
  o << "  \"shave_silicon_load\": \"" << silicon << "\",\n";
  return o.str();
}

std::string build_shave_probe() {
  const std::string dll = find_npu_compiler_dll();
  const bool movi = has_movicompile();
  const std::string signed_elf = find_signed_elf();
  if (dll.empty()) {
    std::ostringstream o;
    o << "  \"shave_compiler_dll\": \"\",\n";
    o << "  \"shave_vcl_version\": \"\",\n";
    o << "  \"shave_elf_inject_export\": false,\n";
    o << "  \"shave_unsigned_ir\": \"not_attempted\",\n";
    o << "  \"shave_movicompile\": " << (movi ? "true" : "false") << ",\n";
    o << "  \"shave_firmware\": [],\n";
    o << "  \"shave_signed_elf\": \"" << json_escape(signed_elf) << "\",\n";
    o << silicon_evidence_fields();
    return o.str();
  }
  HMODULE mod = LoadLibraryExA(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!mod) {
    std::ostringstream o;
    o << "  \"shave_compiler_dll\": \"" << json_escape(dll) << "\",\n";
    o << "  \"shave_vcl_version\": \"\",\n";
    o << "  \"shave_elf_inject_export\": false,\n";
    o << "  \"shave_unsigned_ir\": \"loadlibrary_failed\",\n";
    o << "  \"shave_movicompile\": " << (movi ? "true" : "false") << ",\n";
    o << "  \"shave_firmware\": " << firmware_names(dll) << ",\n";
    o << "  \"shave_signed_elf\": \"" << json_escape(signed_elf) << "\",\n";
    o << silicon_evidence_fields();
    return o.str();
  }
  const bool inject = GetProcAddress(mod, "vclLoadElf") || GetProcAddress(mod, "vclShaveLoad") ||
                      GetProcAddress(mod, "vclExecutableCreateFromElf") ||
                      GetProcAddress(mod, "vclLoadShaveElf") || GetProcAddress(mod, "LoadShaveElf");
  auto getv = reinterpret_cast<vcl_get_version_fn>(GetProcAddress(mod, "vclGetVersion"));
  auto create = reinterpret_cast<vcl_compiler_create_fn>(GetProcAddress(mod, "vclCompilerCreate"));
  auto destroy = reinterpret_cast<vcl_compiler_destroy_fn>(GetProcAddress(mod, "vclCompilerDestroy"));
  auto exec_create = reinterpret_cast<vcl_exec_create_fn>(GetProcAddress(mod, "vclExecutableCreate"));
  auto exec_destroy = reinterpret_cast<vcl_exec_destroy_fn>(GetProcAddress(mod, "vclExecutableDestroy"));
  char verbuf[32] = "";
  uint32_t vrc = 0xffffffffu;
  if (getv) {
    vcl_version_info_t cv{}, pv{};
    vrc = getv(&cv, &pv);
    if (vrc == 0) std::snprintf(verbuf, sizeof(verbuf), "%u.%u", cv.major, cv.minor);
  }
  std::string unsigned_ir = "not_attempted";
  if (getv && create && exec_create) {
    vcl_version_info_t cv{}, pv{};
    if (getv(&cv, &pv) == 0) {
      vcl_compiler_desc_t desc{};
      desc.version = cv;
      desc.debugLevel = kVclLogError;
      vcl_device_desc_t dev{};
      dev.size = sizeof(dev);
      dev.deviceID = 0xAD1D;
      dev.revision = 0xFFFFu;
      dev.tileCount = 0;
      void* compiler = nullptr;
      void* log = nullptr;
      uint32_t crc = vcl_create_seh(create, &desc, &dev, &compiler, &log);
      if (crc != 0 || !compiler) {
        dev.deviceID = 0;
        compiler = nullptr;
        log = nullptr;
        crc = vcl_create_seh(create, &desc, &dev, &compiler, &log);
      }
      if (crc == 0 && compiler) {
        uint8_t elf[64] = {0};
        elf[0] = 0x7f;
        elf[1] = 'E';
        elf[2] = 'L';
        elf[3] = 'F';
        elf[4] = 1;
        elf[5] = 1;
        elf[6] = 1;
        elf[18] = 2;
        vcl_executable_desc_t ed{};
        ed.modelIRData = elf;
        ed.modelIRSize = sizeof(elf);
        ed.options = "";
        ed.optionsSize = 1;
        void* exe = nullptr;
        uint32_t erc = vcl_exec_create_seh(exec_create, compiler, ed, &exe);
        unsigned_ir = vcl_result_name(erc);
        if (exe && exec_destroy) exec_destroy(exe);
        if (!signed_elf.empty()) {
          std::ifstream f(signed_elf, std::ios::binary);
          std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
          if (!bytes.empty()) {
            ed.modelIRData = bytes.data();
            ed.modelIRSize = bytes.size();
            exe = nullptr;
            erc = vcl_exec_create_seh(exec_create, compiler, ed, &exe);
            unsigned_ir.append("+signed_");
            unsigned_ir.append(vcl_result_name(erc));
            if (exe && exec_destroy) exec_destroy(exe);
          }
        }
        if (destroy) destroy(compiler);
      } else {
        unsigned_ir = std::string("compiler_create_") + vcl_result_name(crc);
      }
    }
  } else if (!getv) {
    unsigned_ir = "no_vcl_exports";
  }
  std::ostringstream o;
  o << "  \"shave_compiler_dll\": \"" << json_escape(dll) << "\",\n";
  o << "  \"shave_vcl_version\": \"" << verbuf << "\",\n";
  o << "  \"shave_elf_inject_export\": " << (inject ? "true" : "false") << ",\n";
  o << "  \"shave_unsigned_ir\": \"" << json_escape(unsigned_ir) << "\",\n";
  o << "  \"shave_movicompile\": " << (movi ? "true" : "false") << ",\n";
  o << "  \"shave_firmware\": " << firmware_names(dll) << ",\n";
  o << "  \"shave_signed_elf\": \"" << json_escape(signed_elf) << "\",\n";
  o << silicon_evidence_fields();
  FreeLibrary(mod);
  return o.str();
}
#else
std::string build_shave_probe() {
  return "  \"shave_compiler_dll\": \"\",\n"
         "  \"shave_vcl_version\": \"\",\n"
         "  \"shave_elf_inject_export\": false,\n"
         "  \"shave_unsigned_ir\": \"not_attempted\",\n"
         "  \"shave_movicompile\": false,\n"
         "  \"shave_firmware\": [],\n"
         "  \"shave_signed_elf\": \"\",\n"
         "  \"shave_silicon_load\": \"unsupported_no_compiler\",\n";
}
#endif

}  // namespace

namespace ovvs {
namespace impl {

void append_shave_elf_probe_json(std::ostringstream& o) {
  static std::once_flag once;
  static std::string cached;
  std::call_once(once, [] { cached = build_shave_probe(); });
  o << cached;
}

}  // namespace impl
}  // namespace ovvs
