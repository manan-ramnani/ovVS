#pragma once

#if defined(_WIN32) || defined(_WIN64)
#if defined(IOVS_BUILD)
#define IOVS_API __declspec(dllexport)
#else
#define IOVS_API __declspec(dllimport)
#endif
#else
#if defined(IOVS_BUILD)
#define IOVS_API __attribute__((visibility("default")))
#else
#define IOVS_API
#endif
#endif
