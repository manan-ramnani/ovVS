#pragma once

#if defined(_WIN32) || defined(_WIN64)
#if defined(OVVS_BUILD)
#define OVVS_API __declspec(dllexport)
#else
#define OVVS_API __declspec(dllimport)
#endif
#else
#if defined(OVVS_BUILD)
#define OVVS_API __attribute__((visibility("default")))
#else
#define OVVS_API
#endif
#endif
