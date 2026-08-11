# TVM Quantization: Auto-generating Quantized DNN Kernels

UIUC SRSE Summer 2026 Assignment | Prof. Saša Misailović
Author: Dimitrije Pešić

## Overview

This project analyzes TVM's `relay.quantize` API by quantizing ResNet50
to int8, comparing it against fp32, investigating why the expected
speedup does not materialize on x86, testing sub-byte precisions (int4,
int2, int1), and cross-compiling to aarch64 for validation under QEMU.

See `TVM_quantization_report.pdf` for the full report.

## Key findings

- **x86 native**: int8 is 2.28× *slower* than fp32 — TVM's dispatch logic
  requires both an asymmetric uint8/int8 dtype pairing and a VNNI-aware
  target; neither condition is met by default qconfig + "llvm" target,
  so dispatch falls through to a generic (untuned) kernel.
- **aarch64 (QEMU)**: int8 is 2.07× *faster* than fp32 — the same int8
  graph, on a target whose operator strategy accepts signed int8
  directly, achieves the expected quantization speedup.
- The inversion is direct empirical support for the dispatch hypothesis
  developed by reading TVM source.
- Sub-byte precisions fail at three distinct stages of the TVM pipeline
  (LLVM codegen hang for int4, TIR-level assertion for int2, SimplifyExpr
  assertion for int1), confirming that TVM's quantization framework is
  dtype-agnostic but downstream passes are not.

## Project structure

- `main.tex` / `main.pdf` — full report
- `experiments/` — main benchmark scripts (Steps 1–7 of the assignment)
    - `01_bench_fp32_vs_int8.py` — Steps 1–5: fp32 vs int8 on x86
    - `02_bench_varied_bitwidth.py` — Step 6: sub-byte in logical mode
    - `03_cross_compile_arm.py` — Step 7: cross-compile to aarch64
    - `inspect_compiled.py` — auxiliary: inspect generated LLVM IR/asm
- `diagnostics/` — instrumented probes for native sub-byte failure
  analysis
    - `probe_int{1,2,4}.py` — native int{1,2,4} probes with faulthandler
    - `probe_uint8.py` — uint8/int8 dispatch experiment
    - `failure_analysis.txt` — summary with stack traces and file:line
      citations
- `qemu/` — aarch64 emulation setup and benchmark (Step 9)
    - `README.md` — VM setup and run instructions
    - `boot_virt.sh` — QEMU boot script (virt machine, cortex-a72)
    - `run_bench_in_vm.py` — benchmark script run inside the guest
- `logs/` — captured output from all experiments

## Reproducing the results

See `INSTALL.md` for setup. Then:

```
# x86 benchmarks
python experiments/01_bench_fp32_vs_int8.py 2>&1 | tee logs/fp32_int8_bench.log
python experiments/02_bench_varied_bitwidth.py 2>&1 | tee logs/varied_bitwidth.log

# Sub-byte failure analysis
python diagnostics/probe_int4.py 2>&1 | tee logs/int4_native.log
python diagnostics/probe_int2.py 2>&1 | tee logs/int2_native.log
python diagnostics/probe_int1.py 2>&1 | tee logs/int1_native.log
python diagnostics/probe_uint8.py 2>&1 | tee logs/uint8_bench.log

# Cross-compile to aarch64
python experiments/03_cross_compile_arm.py 2>&1 | tee logs/cross_compile_arm.log

# QEMU benchmark (see qemu/README.md for VM setup first)
```
