#include "ovvs/ovvs.h"

#include <cstdio>
#include <vector>

int main() {
  std::vector<char> buf(32768, 0);
  const ovvsStatus st = ovvsProbeJson(buf.data(), static_cast<int32_t>(buf.size()));
  if (st != OVVS_STATUS_SUCCESS) {
    std::fprintf(stderr, "probe failed: %s\n", ovvsStatusString(st));
    return 1;
  }
  std::fputs(buf.data(), stdout);
  return 0;
}
