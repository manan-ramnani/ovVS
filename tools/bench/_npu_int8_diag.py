"""NPU programming-model probe: dynamic vs weight-stationary, FQ vs f16, infer-only."""
from __future__ import annotations

import time

import numpy as np
import openvino as ov
from openvino import opset8 as op


def timed_infer(compiled, fill, n=20):
    req = compiled.create_infer_request()
    fill(req)
    req.infer()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fill(req)
        req.infer()
    return (time.perf_counter() - t0) * 1000 / n


def set_inputs(req, *arrays):
    for i, a in enumerate(arrays):
        req.set_input_tensor(i, ov.Tensor(a))


def profile(compiled, fill):
    req = compiled.create_infer_request()
    fill(req)
    req.infer()
    rows = []
    try:
        for p in req.profiling_info:
            rows.append(f"{p.node_name}:{p.exec_type}:{p.real_time.total_seconds()*1e3:.3f}ms")
    except Exception as e:
        rows.append(f"no_profile:{e}")
    return rows


def compile_npu(model, extra=None):
    core = ov.Core()
    cfg = {"PERF_COUNT": True}
    if extra:
        cfg.update(extra)
    return core.compile_model(model, "NPU", cfg)


def run():
    rng = np.random.default_rng(0)
    shapes = [(64, 64, 64), (256, 256, 128), (100000, 32, 768)]
    core = ov.Core()
    print("npu", [d for d in core.available_devices if "NPU" in d])
    for M, N, K in shapes:
        print(f"\n=== M={M} N={N} K={K}  (A {M}x{K} @ B {N}x{K}^T) ===")
        A = rng.standard_normal((M, K), dtype=np.float32).astype(np.float32)
        B = rng.standard_normal((N, K), dtype=np.float32).astype(np.float32)
        cases = []

        pA = op.parameter([M, K], ov.Type.f32)
        pB = op.parameter([N, K], ov.Type.f32)
        mm = op.matmul(pA, pB, False, True)
        cases.append(("dyn_f32_x_dyn_f32", ov.Model(mm, [pA, pB]), lambda r: set_inputs(r, A, B)))

        pA = op.parameter([M, K], ov.Type.f32)
        pB = op.parameter([N, K], ov.Type.f32)
        mm = op.convert(op.matmul(op.convert(pA, ov.Type.f16), op.convert(pB, ov.Type.f16), False, True), ov.Type.f32)
        cases.append(("dyn_f16_x_dyn_f16", ov.Model(mm, [pA, pB]), lambda r: set_inputs(r, A, B)))

        pA = op.parameter([M, K], ov.Type.f32)
        mm = op.matmul(pA, op.constant(B), False, True)
        cases.append(("dyn_f32_x_const_f32", ov.Model(mm, [pA]), lambda r: set_inputs(r, A)))

        pA = op.parameter([M, K], ov.Type.f32)
        mm = op.convert(
            op.matmul(op.convert(pA, ov.Type.f16), op.convert(op.constant(B), ov.Type.f16), False, True),
            ov.Type.f32,
        )
        cases.append(("dyn_f16_x_const_f16", ov.Model(mm, [pA]), lambda r: set_inputs(r, A)))

        ra = float(np.max(np.abs(A))) or 1.0
        rb = float(np.max(np.abs(B))) or 1.0
        pA = op.parameter([M, K], ov.Type.f32)
        loA, hiA = op.constant(np.array([-ra], np.float32)), op.constant(np.array([ra], np.float32))
        loB, hiB = op.constant(np.array([-rb], np.float32)), op.constant(np.array([rb], np.float32))
        fqA = op.fake_quantize(pA, loA, hiA, loA, hiA, 256)
        fqB = op.fake_quantize(op.constant(B), loB, hiB, loB, hiB, 256)
        mm = op.matmul(fqA, fqB, False, True)
        cases.append(("fqA_x_fqConstB", ov.Model(mm, [pA]), lambda r: set_inputs(r, A)))

        extras = {
            "fqA_x_fqConstB+dynq": {"NPU_COMPILER_DYNAMIC_QUANTIZATION": True},
            "fqA_x_fqConstB+qdq": {"NPU_QDQ_OPTIMIZATION": True},
        }

        for name, model, fill in cases:
            try:
                compiled = compile_npu(model)
                ms = timed_infer(compiled, fill, n=8 if M >= 10000 else 30)
                prof = profile(compiled, fill)[:8]
                print(f"  {name:24s}  infer {ms:8.3f} ms  {prof[:4]}")
            except Exception as e:
                print(f"  {name:24s}  FAIL {type(e).__name__}: {e}")

        # extra compile flags on the quantized constant-B graph
        pA = op.parameter([M, K], ov.Type.f32)
        loA, hiA = op.constant(np.array([-ra], np.float32)), op.constant(np.array([ra], np.float32))
        loB, hiB = op.constant(np.array([-rb], np.float32)), op.constant(np.array([rb], np.float32))
        fqA = op.fake_quantize(pA, loA, hiA, loA, hiA, 256)
        fqB = op.fake_quantize(op.constant(B), loB, hiB, loB, hiB, 256)
        mm = op.matmul(fqA, fqB, False, True)
        model_q = ov.Model(mm, [pA])
        for tag, cfg in extras.items():
            try:
                compiled = compile_npu(model_q, cfg)
                ms = timed_infer(compiled, lambda r: set_inputs(r, A), n=8 if M >= 10000 else 30)
                prof = profile(compiled, lambda r: set_inputs(r, A))[:4]
                print(f"  {tag:24s}  infer {ms:8.3f} ms  {prof}")
            except Exception as e:
                print(f"  {tag:24s}  FAIL {e}")


if __name__ == "__main__":
    run()
