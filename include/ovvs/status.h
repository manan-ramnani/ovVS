#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ovvsStatus {
  OVVS_STATUS_SUCCESS = 0,
  OVVS_STATUS_ERROR = 1,
  OVVS_STATUS_INVALID_ARGUMENT = 2,
  OVVS_STATUS_UNSUPPORTED = 3,
  OVVS_STATUS_OOM = 4,
  OVVS_STATUS_COMPILE_FAIL = 5,
  OVVS_STATUS_SHAPE_MISMATCH = 6,
  OVVS_STATUS_DEVICE_UNAVAILABLE = 7,
  OVVS_STATUS_IO = 8
} ovvsStatus;

#ifdef __cplusplus
}
#endif
