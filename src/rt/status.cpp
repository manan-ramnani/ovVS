#include "ovvs/ovvs.h"

const char* ovvsGetVersion(void) { return OVVS_VERSION_STRING; }

const char* ovvsStatusString(ovvsStatus status) {
  switch (status) {
    case OVVS_STATUS_SUCCESS:
      return "success";
    case OVVS_STATUS_ERROR:
      return "error";
    case OVVS_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case OVVS_STATUS_UNSUPPORTED:
      return "unsupported";
    case OVVS_STATUS_OOM:
      return "out of memory";
    case OVVS_STATUS_COMPILE_FAIL:
      return "compile fail";
    case OVVS_STATUS_SHAPE_MISMATCH:
      return "shape mismatch";
    case OVVS_STATUS_DEVICE_UNAVAILABLE:
      return "device unavailable";
    case OVVS_STATUS_IO:
      return "io error";
    default:
      return "unknown";
  }
}
