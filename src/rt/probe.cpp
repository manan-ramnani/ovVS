#include "internal.hpp"

#include <cstdlib>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace iovs {
namespace impl {

static std::string cache_home() {
  if (const char* e = std::getenv("IOVS_CACHE_DIR")) return e;
#if defined(_WIN32)
  if (const char* u = std::getenv("USERPROFILE")) return std::string(u) + "\\.cache\\iovs";
  return ".cache/iovs";
#else
  if (const char* h = std::getenv("HOME")) return std::string(h) + "/.cache/iovs";
  return ".cache/iovs";
#endif
}

static std::string cpu_brand() {
#if defined(_WIN32)
  HKEY key;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                    KEY_READ, &key) == ERROR_SUCCESS) {
    char buf[256];
    DWORD sz = sizeof(buf);
    if (RegQueryValueExA(key, "ProcessorNameString", nullptr, nullptr, reinterpret_cast<LPBYTE>(buf),
                         &sz) == ERROR_SUCCESS) {
      RegCloseKey(key);
      return std::string(buf);
    }
    RegCloseKey(key);
  }
#endif
  return "unknown-cpu";
}

static std::string sku_from_cpu(const std::string& brand) {
  if (brand.find("Ultra") != std::string::npos &&
      (brand.find("288V") != std::string::npos || brand.find("258V") != std::string::npos ||
       brand.find("226V") != std::string::npos)) {
    return "lunar-lake";
  }
  if (brand.find("Ultra") != std::string::npos) {
    if (brand.find("265K") != std::string::npos || brand.find("285K") != std::string::npos ||
        brand.find("245K") != std::string::npos || brand.find("200") != std::string::npos) {
      return "arrow-lake";
    }
    return "meteor-lake";
  }
  return "generic-cpu";
}

void probe_fill(ResourcesData& r) {
  r.npu_available = npu_available();
  r.gpu_available = gpu_available();
  r.npu_name = r.npu_available ? "NPU" : "";
  r.gpu_name = r.gpu_available ? "Arc-iGPU" : "";
  r.sku = sku_from_cpu(cpu_brand());
  r.cache_dir = cache_home();
}

std::string probe_json() {
  ResourcesData r;
  probe_fill(r);
  std::ostringstream o;
  o << "{\n";
  o << "  \"sku\": \"" << r.sku << "\",\n";
  o << "  \"cpu\": \"" << cpu_brand() << "\",\n";
  o << "  \"npu_available\": " << (r.npu_available ? "true" : "false") << ",\n";
  o << "  \"gpu_available\": " << (r.gpu_available ? "true" : "false") << ",\n";
  o << "  \"npu_name\": \"" << r.npu_name << "\",\n";
  o << "  \"gpu_name\": \"" << r.gpu_name << "\",\n";
  o << "  \"cache_dir\": \"" << r.cache_dir << "\",\n";
  o << "  \"openvino_built\": "
#if defined(IOVS_WITH_OPENVINO)
    << "true"
#else
    << "false"
#endif
    << ",\n";
  o << "  \"sycl_built\": "
#if defined(IOVS_WITH_SYCL)
    << "true"
#else
    << "false"
#endif
    << ",\n";
  o << "  \"version\": \"" << IOVS_VERSION_STRING << "\"\n";
  o << "}\n";
  return o.str();
}

}  // namespace impl
}  // namespace iovs
