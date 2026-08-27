# NPU GEMM: DPU vs wall clock (Arrow Lake 265K)

Source: `tools/bench/_npu_int8_diag.py` (OpenVINO NPU, `PERF_COUNT`, infer-only after warmup).

MatMul always runs on **DPU**. Host-visible time is DPU plus SHAVE copies plus Level Zero.

| Shape | Graph | DPU MatMul | SHAVE copies (profile) | Wall infer |
|---|---|---|---|---|
| 64×64×64 | two Parameters, f32 | **0.021 ms** | 0.016+0.022 | 0.556 ms |
| 64×64×64 | two Parameters, f16 Convert | 0.021 ms | 0.016+0.021 | 0.573 ms |
| 64×64×64 | A Parameter, B Constant f32 | **0.002 ms** | 0.005 | **8.17 ms** |
| 64×64×64 | FQ A + FQ Constant B (INT8 IR) | 0.009 ms | 0.007 | 6.67 ms |
| 256×256×128 | two Parameters f32 | 0.025 ms | 0.020+0.023 | 0.355 ms |
| 256×256×128 | B Constant | 0.004 ms | 0.005 | **24.5 ms** |
| 1e5×32×768 | two Parameters f32 | **5.26 ms** | **A: 34.8 ms**, B: 0.065, Result: 1.17 | 173 ms |
| 1e5×32×768 | B Constant / FQ | — | — | `ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY` |

CPU f32 64×64×32 was 0.016 ms in `gemm.json`. NPU DPU is in the same µs band; the C API cannot submit in µs.

## C API wall after the feed-path fix

`set_tensor(host*)` made the plugin SHAVE-copy into Level Zero buffers. `ovvsGemm` now `memcpy`s into `get_input_tensor` L0 buffers and reuses one InferRequest (`src/prim/npu/backend_npu.cpp`). Bakeoff (`ovvs_bakeoff`, warmup + 1 timed run):

| Shape | Path | Wall |
|---|---|---|
| 64×128×32 f32 | CPU | 0.015 ms |
| 64×128×32 f32 | NPU C API | 0.78 ms (DPU ~21 µs) |
| 1e5×32×768 f32 | NPU C API | **54 ms** (was 117 ms with `set_tensor`; DPU 5.3 ms) |
| 1e5×32×768 f16 | NPU C API | **50 ms** (was 148 ms) |
| 1e5×32×768 i8 FQ | NPU C API | 209 ms (GPU XMX 122 ms still wins I8) |

Remaining 54 vs 5.3 ms is host memcpy of A (~307 MB) into L0 plus UMD. Closing that needs NPU-allocated tensors as the dataset home, not pointer-sticky skip (k-means mutates centroids in place).

SRAM on this NPU is **4 MB**. `1e5×768` f16 is ~150 MB, so the big operand never lives in scratchpad. `2000×128` f16 is ~0.5 MB and would fit; we still pay SHAVE/L0 to get it there every infer unless it is a compiled Constant, and Constant-B made **wall** worse on 64 and 256 even though DPU got 5–10× faster.

INT8: FakeQuantize+Constant-B DPU 0.009 vs f32 DPU 0.021 (~2×, matches “INT8 double rate”). The compiler will not take raw `si8` MatMul (`tensor<…xsi8>` illegal). Quantized U8/NNCF IR is the contract; two dynamic fp32 Parameters are not.
