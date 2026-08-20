#include "iovs/iovs.h"

#include <cstdio>
#include <vector>

int main() {
  std::vector<char> buf(32768, 0);
  const iovsStatus st = iovsProbeJson(buf.data(), static_cast<int32_t>(buf.size()));
  if (st != IOVS_STATUS_SUCCESS) {
    std::fprintf(stderr, "probe failed: %s\n", iovsStatusString(st));
    return 1;
  }
  std::fputs(buf.data(), stdout);
  return 0;
}
