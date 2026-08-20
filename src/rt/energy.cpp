#include "internal.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

int64_t read_rapl_sysfs() {
#ifndef _WIN32
  const char* paths[] = {"/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                         "/sys/class/powercap/intel-rapl:0/energy_uj"};
  for (const char* p : paths) {
    FILE* f = std::fopen(p, "r");
    if (!f) continue;
    long long v = 0;
    const int n = std::fscanf(f, "%lld", &v);
    std::fclose(f);
    if (n == 1 && v > 0) return static_cast<int64_t>(v);
  }
#endif
  return -1;
}

int64_t read_intel_power_gadget() {
#ifdef _WIN32
  const char* dlls[] = {"EnergyLib64.dll", "IntelEnergyLib.dll",
                        "C:\\Program Files\\Intel\\Power Gadget 3.6\\EnergyLib64.dll",
                        "C:\\Program Files\\Intel\\Power Gadget 3.5\\EnergyLib64.dll"};
  for (const char* name : dlls) {
    HMODULE m = LoadLibraryA(name);
    if (!m) continue;
    using IntelEnergyLibInitialize_t = bool(__cdecl*)();
    using ReadSample_t = bool(__cdecl*)();
    using GetRAPLEnergy_t = bool(__cdecl*)(int, double*);
    auto init = reinterpret_cast<IntelEnergyLibInitialize_t>(GetProcAddress(m, "IntelEnergyLibInitialize"));
    auto sample = reinterpret_cast<ReadSample_t>(GetProcAddress(m, "ReadSample"));
    auto gete = reinterpret_cast<GetRAPLEnergy_t>(GetProcAddress(m, "GetRAPLEnergy"));
    if (init && sample && gete && init() && sample()) {
      double j = 0;
      if (gete(0, &j) && j > 0) return static_cast<int64_t>(j * 1e6);
    }
    FreeLibrary(m);
  }
#endif
  return -1;
}

int64_t read_battery_proxy() {
#ifdef _WIN32
  SYSTEM_POWER_STATUS st{};
  if (GetSystemPowerStatus(&st) && st.BatteryLifePercent <= 100 && st.ACLineStatus != 255) {
    /* Not RAPL; only proves Power APIs work. Do not treat as package energy. */
    (void)st;
  }
#endif
  return -1;
}

}  // namespace

iovsStatus iovsResourcesEnergyUj(iovsResources_t res, int64_t* uj) {
  if (!res || !uj) return IOVS_STATUS_INVALID_ARGUMENT;
  int64_t v = read_rapl_sysfs();
  if (v < 0) v = read_intel_power_gadget();
  if (v < 0) {
    read_battery_proxy();
    *uj = 0;
    return IOVS_STATUS_UNSUPPORTED;
  }
  *uj = v;
  return IOVS_STATUS_SUCCESS;
}
