#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iovsStatus {
  IOVS_STATUS_SUCCESS = 0,
  IOVS_STATUS_ERROR = 1,
  IOVS_STATUS_INVALID_ARGUMENT = 2,
  IOVS_STATUS_UNSUPPORTED = 3,
  IOVS_STATUS_OOM = 4,
  IOVS_STATUS_COMPILE_FAIL = 5,
  IOVS_STATUS_SHAPE_MISMATCH = 6,
  IOVS_STATUS_DEVICE_UNAVAILABLE = 7,
  IOVS_STATUS_IO = 8
} iovsStatus;

#ifdef __cplusplus
}
#endif
