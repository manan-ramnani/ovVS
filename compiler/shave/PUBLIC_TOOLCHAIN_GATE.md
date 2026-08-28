# Public `npu_compiler` gate

`public_toolchain_gate.ps1` is a fail-closed feasibility gate for an ovVS
Activation SHAVE kernel. It never downloads a toolchain, edits an OpenVINO
installation, patches the source checkouts, or treats host code as NPU code.
An optional external verifier is recorded but its side effects are not audited.

The order is deliberate:

1. Run the pinned `MatMulWithDivide` fixture, which explicitly requests
   `NPU_COMPILER_TYPE=PLUGIN`, and require a real NPU functional-test pass.
   The gate requires clean pinned trees, retained verbose build logs, both
   CMake caches, exactly one core compiler library, and an NPU log naming that
   exact hashed library. Missing provenance stops the baseline.
2. Run an existing NPU3720 ReferenceSW test. At this pin the test neither
   explicitly selects Compiler-In-Plugin nor enables profiling, so it is kept
   only as stock SW-graph evidence and recorded as **not custom**.
3. Check the public custom-kernel chain: exact compiler/OpenVINO commits,
   kernel source and descriptor, MoviTools compiler/linker/runtime libraries,
   a newly built `3720xx` ELF, compiler lowering/registration, and an on-device
   oracle/profiling run. Caller-printed markers are retained but never authorize
   `PASS`; that needs a checked-in bounded verifier which parses raw evidence.

## Current public boundary

At compiler commit `6761af885b8ff54ddf0da5bf8ad44e30746b2f62`,
`validation/openvino_config.json` pins OpenVINO
`4089686065a245d648cdd2b99c31884f53cb7a5e`. The public tree includes Intel
prebuilt ActShave ELFs, but it tracks no `sw_runtime_kernels/kernels/src/*` or
`descrip/*` files. Enabling `ENABLE_SHAVE_BINARIES_BUILD` is destructive in
that state: the public CMake removes every prebuilt without a descriptor. Its
downloader reads
`artifacts/vpuip_2/revisions.json`; neither that path nor the README's
`artifacts/vpuip2/revisions.json` is public. A forced MoviTools tree must contain:

```text
win64/bin/moviCompile.exe
win64/bin/moviLLD.exe
common/moviCompile/lib/37xxxx/mlibm.a
common/moviCompile/lib/37xxxx/mlibc_lite.a
common/moviCompile/lib/37xxxx/mlibcrt.a
```

There is no separate public user-signing command in this source path. MoviTools
produces an ELF32 kernel; CMake embeds its bytes into the Compiler-In-Plugin,
which emits the native graph submitted to the driver. Firmware acceptance is
therefore proven only by executing the resulting graph. A loose ELF, an
`add_extension()` op, a renamed Intel prebuilt ELF, or `compiler/shave/*.c`
linked into `libovvs` does not pass.

The 2026-08-29 read-only machine audit stopped at this boundary:

- an ignored shallow clone now exists at `compiler/npu_compiler` in this
  worktree at the pinned compiler commit;
- there was no pinned OpenVINO source/developer build and no compiler build
  directory, so the unchanged Compiler-In-Plugin graph was not runnable;
- the public commit tracks 770 prebuilt ELFs, zero kernel sources, and zero
  descriptors; both MoviTools manifest paths are absent;
- `moviCompile` and `moviLLD` were absent from `PATH` and the inspected Intel
  installation roots;
- DriverStore did contain a production compiler with product version
  `2026.3.0-1-4089686065a-06a684dde5a`; matching the OpenVINO pin does not make
  that production DLL a local Compiler-In-Plugin build, so the gate rejects it;
- `activation_sigmoid.3720xx.elf` is a 1,712-byte Intel prebuilt with SHA-256
  `B5E06C1EE029CC4CBA74AB91C7B469607700A8D891630AEB447B181462C32B3A`.

Accordingly, the current `BASELINE_BLOCKED` report also retains independent
`UNSUPPORTED_PUBLIC_TOOLCHAIN` evidence for MoviTools and the unpublished
source/descriptor set. This is not a custom-kernel success.

## Isolated build commands

Run these only in new source/build directories. Do not point them at the working
OpenVINO installation or its cache. The developer preset also requires
`ccache`, as documented by the pinned compiler tree.

```bat
rem OpenVINO source must be at 4089686065a245d648cdd2b99c31884f53cb7a5e
set OPENVINO_HOME=C:\isolated\openvino
set NPU_COMPILER_HOME=C:\isolated\npu_compiler
set BUILD_LOG_ROOT=C:\isolated\build-logs
set CONFIG=RelWithDebInfo
if not exist %BUILD_LOG_ROOT% mkdir %BUILD_LOG_ROOT%

cd /d %OPENVINO_HOME%
git rev-parse HEAD > %BUILD_LOG_ROOT%\openvino-build.log
echo targets=openvino_intel_npu_plugin >> %BUILD_LOG_ROOT%\openvino-build.log
cmake -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% -DENABLE_INTEL_NPU=ON -DENABLE_PLUGINS_XML=ON -DENABLE_INTEL_NPU_COMPILER=OFF -DENABLE_DEBUG_CAPS=ON -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -B build-x86_64\%CONFIG%
cmake --build build-x86_64\%CONFIG% --clean-first --verbose --target ov_dev_targets openvino_intel_npu_plugin compile_tool >> %BUILD_LOG_ROOT%\openvino-build.log 2>&1

cd /d %NPU_COMPILER_HOME%
set OPENVINO_HOME=C:\isolated\openvino
git rev-parse HEAD > %BUILD_LOG_ROOT%\compiler-build.log
echo target=openvino_intel_npu_compiler >> %BUILD_LOG_ROOT%\compiler-build.log
cmake --preset developer-build-relwithdebinfo -DENABLE_SHAVE_BINARIES_BUILD=OFF .
cmake --build build-x86_64\RelWithDebInfo --clean-first --verbose --target npuFuncTests >> %BUILD_LOG_ROOT%\compiler-build.log 2>&1
```

The unchanged build must pass before attempting
`-DENABLE_SHAVE_BINARIES_BUILD=ON`. On the audited public tree that second
configuration is unsupported until the missing source/descriptor and MoviTools
inputs are supplied through an authorized isolated toolchain.

Example gate invocation after the unchanged build exists:

```powershell
pwsh -File .\compiler\shave\public_toolchain_gate.ps1 `
  -CompilerRoot C:\isolated\npu_compiler `
  -OpenVinoRoot C:\isolated\openvino `
  -OpenVinoBuildLog C:\isolated\build-logs\openvino-build.log `
  -CompilerBuildLog C:\isolated\build-logs\compiler-build.log `
  -EvidenceDir C:\isolated\evidence\ovvs-shave-gate
```

Exit codes are stable: `10` unchanged-graph baseline blocked, `20` public
custom-kernel toolchain unsupported, `21` custom inputs/verification required,
`30` custom verification failed or remains self-attested, and `40` invalid
invocation. Exit `0` is reserved for a future checked-in verifier; this gate
currently has no path which emits it. A non-empty evidence directory is rejected
so earlier failed or unsupported evidence is never overwritten.

## Evidence report template

```text
Compiler commit:
OpenVINO commit:
NPU driver/compiler version:
Baseline test/filter and pass count:
Built-in ActShave test/filter (classified not custom):
MoviTools root and executable hashes:
Kernel source + descriptor hashes:
Custom ELF name + SHA-256:
Compiler registration/lowering commit:
Graph compile cold time:
Warm repetitions:
CPU oracle comparison:
Profiling attribution (must say Shave for the custom entry):
Result: BASELINE_BLOCKED / UNSUPPORTED_PUBLIC_TOOLCHAIN / CUSTOM_INPUTS_REQUIRED / READY_FOR_CUSTOM_VERIFICATION / CUSTOM_VERIFICATION_FAILED / CUSTOM_VERIFICATION_UNPROVEN
Blockers retained verbatim:
```
