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

| Shape | Path | Wall (ms / ms_hot) |
|---|---|---|
| 64×128×32 f32 | CPU oneMKL | **0.018 / 0.006** |
| 64×128×32 f32 | NPU C API | 0.94 / 0.57 |
| 1e5×32×768 f32 | CPU oneMKL | **18 / 20** |
| 1e5×32×768 f32 | NPU C API | 45 / 53 (DPU 5.3) |
| 32×1e5×768 f32 | CPU oneMKL | **18 / 26** |
| 32×1e5×768 f32 | NPU C API | 36 / 35 |
| 1e5×32×768 f16 | NPU C API | 39 / 42 |
| 1e5×32×768 i8 FQ | NPU C API | 231 / 223 |

Fingerprint-sticky skips host memcpy of unchanged ≥64 KiB operands. The remaining NPU wall vs DPU is **device DMA of the Parameter into SRAM**, not a second plugin copy. oneMKL `cblas_sgemm` reads the same DRAM faster than that DMA. AUTO GEMM is CPU on this SKU. The installed OpenVINO 2025.3 headers expose `create_l0_host_tensor`, but remote host memory may only change application copies, buffer ownership, or cold-path cost; it does not make a large operand scratchpad-resident or prove a steady-state wall reduction. B7 must measure those effects separately. A compiled Constant also remains rejected unless end-to-end wall improves.

SRAM on this NPU is **4 MB**. `1e5×768` f16 is ~150 MB, so the big operand never lives in scratchpad. `2000×128` f16 is ~0.5 MB and would fit; we still pay SHAVE/L0 to get it there every infer unless it is a compiled Constant, and Constant-B made **wall** worse on 64 and 256 even though DPU got 5–10× faster.

INT8: FakeQuantize+Constant-B DPU 0.009 vs f32 DPU 0.021 (~2×, matches “INT8 double rate”). The compiler will not take raw `si8` MatMul (`tensor<…xsi8>` illegal). Quantized U8/NNCF IR is the contract; two dynamic fp32 Parameters are not.
