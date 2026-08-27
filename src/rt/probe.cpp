#include "internal.hpp"

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace ovvs {
namespace impl {

static std::string cache_home() {
  if (const char* e = std::getenv("OVVS_CACHE_DIR")) return e;
#if defined(_WIN32)
  if (const char* u = std::getenv("USERPROFILE")) return std::string(u) + "\\.cache\\ovvs";
  return ".cache/ovvs";
#else
  if (const char* h = std::getenv("HOME")) return std::string(h) + "/.cache/ovvs";
  return ".cache/ovvs";
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

static std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static float json_run_ms(const std::string& s, const char* requested) {
  const std::string key = std::string("\"requested\": \"") + requested + "\"";
  const auto p = s.find(key);
  if (p == std::string::npos) return -1.f;
  const auto m = s.find("\"ms\":", p);
  if (m == std::string::npos || m > p + 180) return -1.f;
  return static_cast<float>(std::atof(s.c_str() + m + 5));
}

static std::vector<std::string> table_candidates(const std::string& sku) {
  std::vector<std::string> out;
  if (const char* e = std::getenv("OVVS_TABLES")) {
    out.push_back(std::string(e) + "/" + sku + "/gemm_large.json");
    out.push_back(std::string(e) + "/gemm_large.json");
  }
  out.push_back(std::string("tables/") + sku + "/gemm_large.json");
#ifdef _WIN32
  char mod[MAX_PATH] = {0};
  if (GetModuleFileNameA(nullptr, mod, MAX_PATH)) {
    std::string dir(mod);
    const auto slash = dir.find_last_of("\\/");
    if (slash != std::string::npos) dir.resize(slash);
    out.push_back(dir + "/../../tables/" + sku + "/gemm_large.json");
    out.push_back(dir + "/../../../tables/" + sku + "/gemm_large.json");
  }
#endif
  return out;
}

static void load_large_gemm_table(ResourcesData& r) {
  for (const auto& path : table_candidates(r.sku)) {
    const std::string s = read_file(path);
    if (s.empty()) continue;
    const float cpu = json_run_ms(s, "cpu");
    const float npu = json_run_ms(s, "npu");
    const float gpu = json_run_ms(s, "gpu");
    float best = 1e30f;
    ovvsDevice win = OVVS_DEVICE_AUTO;
    if (cpu > 0 && cpu < best) {
      best = cpu;
      win = OVVS_DEVICE_CPU;
    }
    if (npu > 0 && npu < best) {
      best = npu;
      win = OVVS_DEVICE_NPU;
    }
    if (gpu > 0 && gpu < best) {
      best = gpu;
      win = OVVS_DEVICE_GPU;
    }
    if (win != OVVS_DEVICE_AUTO) {
      r.large_gemm_winner = win;
      return;
    }
  }
}

void probe_fill(ResourcesData& r) {
  r.npu_available = npu_available();
  r.gpu_available = gpu_available();
  r.npu_name = r.npu_available ? "NPU" : "";
  r.gpu_name = r.gpu_available ? "Arc-iGPU" : "";
  r.sku = sku_from_cpu(cpu_brand());
  r.cache_dir = cache_home();
  load_large_gemm_table(r);
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
#if defined(OVVS_WITH_OPENVINO)
    << "true"
#else
    << "false"
#endif
    << ",\n";
  o << "  \"sycl_built\": "
#if defined(OVVS_WITH_SYCL)
    << "true"
#else
    << "false"
#endif
    << ",\n";
  append_energy_probe_json(o);
  append_shave_elf_probe_json(o);
  append_lowbit_probe_json(o);
  o << "  \"version\": \"" << OVVS_VERSION_STRING << "\"\n";
  o << "}\n";
  return o.str();
}

}  // namespace impl
}  // namespace ovvs
