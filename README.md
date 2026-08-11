# TVM Quantization: Why INT8 ResNet50 Ran 2.28× Slower Than FP32 on x86

A compiler-level investigation of an unexpected benchmark result:
post-training int8 quantization of ResNet50 through TVM's
`relay.quantize` made inference *slower* on x86. This repository traces
the cause through TVM's operator-strategy dispatch and validates the
explanation by cross-compiling the same quantized graph to AArch64.

## The result

ResNet50 (torchvision, batch 1), TVM v0.15.0, `opt_level=3`, default
untuned schedules, 100 timed runs per configuration:

| Platform | fp32 mean | int8 mean | int8 vs fp32 |
|---|---:|---:|---|
| x86-64 (native, plain `llvm` target) | 166.6 ms | 379.9 ms | **2.28× slower** |
| AArch64 (QEMU `virt`, Cortex-A72) | 9550.5 ms | 4619.2 ms | **2.07× faster** |

The identical int8 Relay graph that loses 2.28× on x86 wins 2.07× when
cross-compiled to AArch64. The missing speedup is a property of the x86
compile path — not of int8 quantization itself.

*Research artifact from the UIUC SRSE Summer 2026 program (advisor:
Prof. Saša Misailović). Author: Dimitrije Pešić. Full write-up:
[`TVM_quantization_report.pdf`](TVM_quantization_report.pdf).*

## Root cause: x86 operator-strategy dispatch

TVM's x86 conv2d strategy (`conv2d_NCHWc_strategy_cpu` in
`python/tvm/relay/op/strategy/x86.py`) only selects the int8-optimized
kernel (`conv2d_NCHWc_int8.x86`) when `is_int8_hw_support` (in
`python/tvm/topi/x86/conv2d_int8.py`) passes **both** of its checks:

1. an asymmetric dtype pairing — `uint8` data × `int8` weights —
   inherited from Intel's `VPDPBUSD` instruction (AVX-512 VNNI), which
   computes an unsigned-byte × signed-byte dot product into an int32
   accumulator; and
2. a compilation target whose `-mcpu` advertises VNNI
   (e.g. `-mcpu=cascadelake`).

The default `relay.quantize` qconfig produces signed int8 for both data
and weights, and the benchmark target was plain `"llvm"`, so both checks
fail and every convolution dispatches to the generic
`conv2d_NCHWc.x86` kernel. The int8 graph then pays the requantization
overhead — each fp32 `conv2d → add → relu` block becomes a 13-operation
chain with `right_shift`, `clip`, `cast`, and `multiply` — with no
compute-throughput gain to amortize it.

Three independent checks support the diagnosis:

- **Dispatch logs** (`diagnostics/probe_uint8.py`): switching only the
  dtype pairing to uint8 × int8 still routes every convolution to the
  generic kernel, because the plain `llvm` target fails check 2. In a
  single controlled session, uint8/int8 measured 351.8 ms vs 396.1 ms
  for signed int8/int8 — both still far above the ~166 ms fp32
  baseline. The dtype pairing alone is not sufficient.
- **Generated code** (`experiments/inspect_compiled.py`): the int8
  build's assembly contains no `vpdpbusd` instructions, confirming the
  VNNI path was never emitted.
- **Cross-architecture** (`experiments/03_cross_compile_arm.py` +
  `qemu/`): compiled for `-mtriple=aarch64-linux-gnu
  -mcpu=cortex-a72`, the same int8 graph runs 2.07× faster than fp32
  under QEMU. TVM's ARM operator strategy accepts the signed
  int8 / int8 pairing directly (we did not trace the ARM dispatch in
  source to the same depth as x86; the measured inversion is the
  empirical evidence). The result was additionally validated on QEMU's
  actual `raspi4b` board model, where the int8 inference completed with
  the same output argmax as the `virt` run
  (`logs/raspi4b_resnet50_int8.log`).

## Sub-byte precisions: three distinct failure stages

`relay.quantize` accepts arbitrary bit widths, but native sub-byte
dtypes fail at three different stages of the pipeline (full stack traces
and file:line analysis in `diagnostics/failure_analysis.txt`):

| Precision | Stage reached | Failure |
|---|---|---|
| int4 | `relay.build()` | Never terminates: CPU-bound in a C++ pass for 29+ minutes with no diagnostic; the build stalls *before* reaching LLVM codegen's `bits >= 8` buffer-access check |
| int2 | `quantize.quantize()` | Clean `InternalError`: calibration JIT-compiles small subgraphs, and the TIR buffer-binding pass (`ArgBinder::BindDLTensor` → `GetVectorBytes`) rejects the 2-bit storage dtype (`data_bits % 8 == 0` check) |
| int1 | `relay.build()` | Clean `InternalError` in Relay's `SimplifyExpr` pass: validating a clip's bounds computes `min_value(int1)`, constructing `IntImm(int1, -1)` — outside the [0, 1] range TVM assigns to `Int(1)` |

The quantization pass itself is dtype-agnostic; downstream passes are
not, and each imposes its own dtype assumptions with no unified
validation layer.

A control experiment ("logical" sub-byte mode,
`experiments/02_bench_varied_bitwidth.py`) sets `nbit < 8` while keeping
int8 storage: every width then builds and runs, confirming the failures
above are downstream dtype-handling issues rather than quantization
logic. nbit=4 and nbit=2 land near the int8 baseline (419.1 ms and
403.7 ms); at nbit=1 latency collapses to 9.8 ms because the quantized
graph becomes degenerate and largely constant-folds away — an artifact
of algebraic simplification, not a faster kernel.

## Additional measurements

- **Weight compression**: serialized int8 `.params` is 30.4 MB vs
  191.5 MB for fp32 (6.30×) — the storage benefit holds regardless of
  the compute-side dispatch outcome.
- **raspi4b scalar microbenchmark** (`qemu/microbench.c`): scalar
  integer MAC throughput is ~2.6–2.7× fp32 under QEMU TCG — consistent
  in direction with the ResNet50 inversion, and a property of TCG
  translation costs rather than of native Pi 4 silicon.
- **fp32 on raspi4b does not fit**: the machine model's RAM is fixed at
  2 GiB and the ~385 MB fp32 artifact triple overflows the initramfs
  unpack (`logs/raspi4b_resnet50_fp32_oom.log`), so fp32 was measured
  on the `virt` machine only.

## Scope and caveats

- All schedules are untuned by design. The findings concern dispatch
  structure and IR shape, not schedule quality; AutoTVM/MetaSchedule
  would change absolute numbers. Tuning under QEMU would also be
  meaningless, since the emulator's timing signal does not reflect real
  hardware.
- QEMU numbers are TCG software emulation (no KVM): absolute latencies
  overestimate native Cortex-A72 by a large factor. Only the
  fp32-vs-int8 ratio is meaningful, since both configurations pay the
  same emulation cost.
- x86 measurements were taken on a WSL2 host in single sessions;
  cross-session numbers are compared only qualitatively.

## Repository layout

```
main.tex                     LaTeX source of the report
TVM_quantization_report.pdf  compiled report
INSTALL.md                   toolchain setup (TVM v0.15.0 from source)
experiments/
  01_bench_fp32_vs_int8.py     x86 fp32 vs int8 benchmark
  02_bench_varied_bitwidth.py  logical-mode sub-byte benchmark
  03_cross_compile_arm.py      cross-compile both models to AArch64
  inspect_compiled.py          LLVM IR / assembly / graph inspection
diagnostics/
  probe_int4.py, probe_int2.py, probe_int1.py
                               instrumented native sub-byte probes
                               (faulthandler, periodic stack dumps)
  probe_uint8.py               uint8 x int8 dispatch experiment
  failure_analysis.txt         failure sites with stack traces
qemu/
  README.md                    both QEMU setups, step by step
  boot_virt.sh                 virt-machine boot script
  run_bench_in_vm.py           benchmark run inside the virt guest
  microbench.c                 raspi4b PID-1 MAC microbenchmark
  resnet50_bench.cpp           raspi4b PID-1 TVM C++ runtime benchmark
logs/                          captured output of every experiment
```

## Reproducing the results

See [`INSTALL.md`](INSTALL.md) for setup (TVM v0.15.0 built from
source; the reported x86 numbers were measured on an x86-64 WSL2 host
with Python 3.10). Then, from the repository root:

```bash
# x86 benchmarks
python experiments/01_bench_fp32_vs_int8.py 2>&1 | tee logs/fp32_int8_bench.log
python experiments/02_bench_varied_bitwidth.py 2>&1 | tee logs/varied_bitwidth.log
python experiments/inspect_compiled.py 2>&1 | tee logs/inspect_compiled.log

# Native sub-byte failure probes
# (probe_int4 hangs by design: observe the periodic stack dumps, then kill it)
python diagnostics/probe_int4.py 2>&1 | tee logs/int4_native.log
python diagnostics/probe_int2.py 2>&1 | tee logs/int2_native.log
python diagnostics/probe_int1.py 2>&1 | tee logs/int1_native.log
python diagnostics/probe_uint8.py 2>&1 | tee logs/uint8_bench.log

# Cross-compile to AArch64 (requires gcc-aarch64-linux-gnu)
python experiments/03_cross_compile_arm.py 2>&1 | tee logs/cross_compile_arm.log

# QEMU benchmarks: see qemu/README.md for the virt and raspi4b setups
```

Exact timings vary with the host CPU; the qualitative results (the x86
slowdown, the AArch64 inversion, and the three sub-byte failure sites)
are what the investigation establishes.
