#include "internal.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <winioctl.h>
#include <setupapi.h>
#include <initguid.h>
#include <emi.h>
#include <pdh.h>
#include <mutex>
#endif

namespace {

const char* g_energy_source = "unsupported";
char g_energy_channel[64] = "";

int64_t pwh_to_uj(uint64_t pwh) {
  /* EMI AbsoluteEnergy is picowatt-hours. 1 pWh = 3.6e-9 J = 0.0036 µJ. */
  if (pwh > static_cast<uint64_t>(INT64_MAX / 36)) {
    return static_cast<int64_t>((pwh / 10000ull) * 36ull);
  }
  return static_cast<int64_t>((pwh * 36ull) / 10000ull);
}

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
    if (n == 1 && v > 0) {
      g_energy_source = "linux-rapl";
      std::snprintf(g_energy_channel, sizeof(g_energy_channel), "package0");
      return static_cast<int64_t>(v);
    }
  }
#endif
  return -1;
}

#ifdef _WIN32
struct EmiState {
  HANDLE device = INVALID_HANDLE_VALUE;
  int pkg_channel = 0;
  int channel_count = 0;
  bool ok = false;
};

EmiState g_emi;
std::once_flag g_emi_once;

static bool channel_is_pkg(const wchar_t* name) {
  /* Names are RAPL_Package0_{PKG,PP0,PP1,DRAM}. "Package" matches every channel. */
  return name && wcsstr(name, L"PKG") != nullptr;
}

static bool channel_is_dram(const wchar_t* name) {
  return name && wcsstr(name, L"DRAM") != nullptr;
}

static void emi_init_once() {
  HDEVINFO dev = SetupDiGetClassDevsW(const_cast<GUID*>(&GUID_DEVICE_ENERGY_METER), nullptr, nullptr,
                                      DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
  if (dev == INVALID_HANDLE_VALUE) return;
  SP_DEVICE_INTERFACE_DATA iface{};
  iface.cbSize = sizeof(iface);
  for (DWORD i = 0; SetupDiEnumDeviceInterfaces(dev, nullptr, &GUID_DEVICE_ENERGY_METER, i, &iface); ++i) {
    DWORD need = 0;
    SetupDiGetDeviceInterfaceDetailW(dev, &iface, nullptr, 0, &need, nullptr);
    if (need == 0) continue;
    std::vector<uint8_t> buf(need);
    auto* detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(buf.data());
    detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
    if (!SetupDiGetDeviceInterfaceDetailW(dev, &iface, detail, need, nullptr, nullptr)) continue;
    HANDLE h = CreateFileW(detail->DevicePath, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) continue;
    EMI_VERSION ver{};
    DWORD ret = 0;
    if (!DeviceIoControl(h, IOCTL_EMI_GET_VERSION, nullptr, 0, &ver, sizeof(ver), &ret, nullptr)) {
      CloseHandle(h);
      continue;
    }
    EMI_METADATA_SIZE msz{};
    if (!DeviceIoControl(h, IOCTL_EMI_GET_METADATA_SIZE, nullptr, 0, &msz, sizeof(msz), &ret, nullptr) ||
        msz.MetadataSize == 0) {
      CloseHandle(h);
      continue;
    }
    std::vector<uint8_t> meta(msz.MetadataSize);
    if (!DeviceIoControl(h, IOCTL_EMI_GET_METADATA, nullptr, 0, meta.data(), static_cast<DWORD>(meta.size()),
                         &ret, nullptr)) {
      CloseHandle(h);
      continue;
    }
    int nch = 1;
    int pkg = 0;
    char chname[64] = "channel0";
    if (ver.EmiVersion >= EMI_VERSION_V2 && meta.size() >= sizeof(EMI_METADATA_V2)) {
      auto* m = reinterpret_cast<EMI_METADATA_V2*>(meta.data());
      nch = static_cast<int>(m->ChannelCount);
      if (nch <= 0) nch = 1;
      auto* ch = m->Channels;
      uint8_t* end = meta.data() + meta.size();
      int fallback = 0;
      pkg = -1;
      std::vector<const wchar_t*> names(static_cast<size_t>(nch), nullptr);
      for (int c = 0; c < nch && reinterpret_cast<uint8_t*>(ch) + sizeof(USHORT) * 2 <= end; ++c) {
        names[static_cast<size_t>(c)] = ch->ChannelName;
        if (channel_is_pkg(ch->ChannelName)) pkg = c;
        else if (fallback == 0 && c > 0 && !channel_is_dram(ch->ChannelName)) fallback = c;
        else if (c == 0 && !channel_is_dram(ch->ChannelName)) fallback = 0;
        ch = EMI_CHANNEL_V2_NEXT_CHANNEL(ch);
      }
      if (pkg < 0) pkg = fallback;
      const wchar_t* wn = names[static_cast<size_t>(pkg)];
      if (wn) WideCharToMultiByte(CP_UTF8, 0, wn, -1, chname, sizeof(chname), nullptr, nullptr);
    } else if (meta.size() >= sizeof(EMI_METADATA_V1)) {
      auto* m = reinterpret_cast<EMI_METADATA_V1*>(meta.data());
      WideCharToMultiByte(CP_UTF8, 0, m->MeteredHardwareName, -1, chname, sizeof(chname), nullptr, nullptr);
    }
    g_emi.device = h;
    g_emi.pkg_channel = pkg < 0 ? 0 : pkg;
    g_emi.channel_count = nch;
    g_emi.ok = true;
    std::snprintf(g_energy_channel, sizeof(g_energy_channel), "%s", chname);
    break;
  }
  SetupDiDestroyDeviceInfoList(dev);
}

int64_t read_emi_intelppm() {
  std::call_once(g_emi_once, emi_init_once);
  if (!g_emi.ok || g_emi.device == INVALID_HANDLE_VALUE) return -1;
  if (g_emi.channel_count <= 0) return -1;
  std::vector<EMI_CHANNEL_MEASUREMENT_DATA> meas(static_cast<size_t>(g_emi.channel_count));
  DWORD ret = 0;
  if (!DeviceIoControl(g_emi.device, IOCTL_EMI_GET_MEASUREMENT, nullptr, 0, meas.data(),
                       static_cast<DWORD>(meas.size() * sizeof(EMI_CHANNEL_MEASUREMENT_DATA)), &ret, nullptr)) {
    return -1;
  }
  const int ch = g_emi.pkg_channel;
  if (ch < 0 || ch >= g_emi.channel_count) return -1;
  const uint64_t pwh = meas[static_cast<size_t>(ch)].AbsoluteEnergy;
  if (pwh == 0) return -1;
  g_energy_source = "emi-intelppm";
  return pwh_to_uj(pwh);
}

int64_t read_pdh_energy_meter() {
  PDH_HQUERY query = nullptr;
  if (PdhOpenQueryA(nullptr, 0, &query) != ERROR_SUCCESS) return -1;
  PDH_HCOUNTER counter = nullptr;
  const char* paths[] = {"\\Energy Meter(RAPL_Package0_PKG)\\Energy",
                         "\\Energy Meter(RAPL_Package0_PP0)\\Energy"};
  PDH_STATUS st = 0xFFFFFFFF;
  const char* used = nullptr;
  for (const char* p : paths) {
    st = PdhAddEnglishCounterA(query, p, 0, &counter);
    if (st == ERROR_SUCCESS) {
      used = p;
      break;
    }
  }
  int64_t uj = -1;
  if (st == ERROR_SUCCESS && PdhCollectQueryData(query) == ERROR_SUCCESS) {
    PDH_FMT_COUNTERVALUE v{};
    if (PdhGetFormattedCounterValue(counter, PDH_FMT_LARGE, nullptr, &v) == ERROR_SUCCESS && v.largeValue > 0) {
      uj = pwh_to_uj(static_cast<uint64_t>(v.largeValue));
      g_energy_source = "pdh-energy-meter";
      std::snprintf(g_energy_channel, sizeof(g_energy_channel), "%s",
                    std::strstr(used, "PP0") ? "RAPL_Package0_PP0" : "RAPL_Package0_PKG");
    }
  }
  PdhCloseQuery(query);
  return uj;
}
#endif

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
      if (gete(0, &j) && j > 0) {
        g_energy_source = "power-gadget";
        std::snprintf(g_energy_channel, sizeof(g_energy_channel), "RAPL_package");
        const int64_t uj = static_cast<int64_t>(j * 1e6);
        FreeLibrary(m);
        return uj;
      }
    }
    FreeLibrary(m);
  }
#endif
  return -1;
}

}  // namespace

namespace ovvs {
namespace impl {

void append_energy_probe_json(std::ostringstream& o) {
  int64_t uj = 0;
  const ovvsStatus st = ovvsResourcesEnergyUj(nullptr, &uj);
  o << "  \"energy_source\": \"" << g_energy_source << "\",\n";
  o << "  \"energy_channel\": \"" << g_energy_channel << "\",\n";
  o << "  \"energy_uj\": " << (st == OVVS_STATUS_SUCCESS ? uj : 0) << ",\n";
  o << "  \"energy_status\": \"" << ovvsStatusString(st) << "\",\n";
}

}  // namespace impl
}  // namespace ovvs

ovvsStatus ovvsResourcesEnergyUj(ovvsResources_t res, int64_t* uj) {
  (void)res;
  if (!uj) return OVVS_STATUS_INVALID_ARGUMENT;
  int64_t v = read_rapl_sysfs();
#ifdef _WIN32
  if (v < 0) v = read_emi_intelppm();
  if (v < 0) v = read_pdh_energy_meter();
#endif
  if (v < 0) v = read_intel_power_gadget();
  if (v < 0) {
    g_energy_source = "unsupported";
    *uj = 0;
    return OVVS_STATUS_UNSUPPORTED;
  }
  *uj = v;
  return OVVS_STATUS_SUCCESS;
}
