#include "iovs/iovs.h"

const char* iovsGetVersion(void) { return IOVS_VERSION_STRING; }

const char* iovsStatusString(iovsStatus status) {
  switch (status) {
    case IOVS_STATUS_SUCCESS:
      return "success";
    case IOVS_STATUS_ERROR:
      return "error";
    case IOVS_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case IOVS_STATUS_UNSUPPORTED:
      return "unsupported";
    case IOVS_STATUS_OOM:
      return "out of memory";
    case IOVS_STATUS_COMPILE_FAIL:
      return "compile fail";
    case IOVS_STATUS_SHAPE_MISMATCH:
      return "shape mismatch";
    case IOVS_STATUS_DEVICE_UNAVAILABLE:
      return "device unavailable";
    case IOVS_STATUS_IO:
      return "io error";
    default:
      return "unknown";
  }
}
