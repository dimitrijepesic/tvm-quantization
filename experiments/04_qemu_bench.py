## benchmark fp32 vs int8 resnet50 inference on raspberry pi emulated via qemu
## this script runs inside the qemu guest (aarch64 raspberry pi 3b)
## models are cross-compiled on host using 03_cross_compile_arm.py

import sys
import os
import time
import numpy as np

# add tvm python path (on the qemu guest filesystem)
sys.path.insert(0, "/home/pi/tvm/python")
os.environ["TVM_LIBRARY_PATH"] = "/home/pi/tvm/lib"

import tvm
from tvm.contrib import graph_executor

MODEL_DIR = "/home/pi/models"
INPUT_SHAPE = (1, 3, 224, 224)
NUM_WARMUP = 1
BENCH_NUMBER = 2
BENCH_REPEAT = 3


def load_and_benchmark(model_prefix, label):
    # load a tvm model and benchmark it
    so_path = os.path.join(MODEL_DIR, f"{model_prefix}.so")
    json_path = os.path.join(MODEL_DIR, f"{model_prefix}.json")
    params_path = os.path.join(MODEL_DIR, f"{model_prefix}.params")

    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"{'='*60}")

    for p in [so_path, json_path, params_path]:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  {os.path.basename(p)}: {size_mb:.1f} MB")

    # load model
    print("Loading model...", flush=True)
    t0 = time.time()
    lib = tvm.runtime.load_module(so_path)
    with open(json_path, "r") as f:
        graph_json = f.read()
    with open(params_path, "rb") as f:
        params_bytes = f.read()
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")

    # create graph executor
    dev = tvm.cpu(0)
    module = graph_executor.create(graph_json, lib, dev)
    module.load_params(params_bytes)

    # prepare input
    input_data = np.random.randn(*INPUT_SHAPE).astype("float32")
    module.set_input("input0", tvm.nd.array(input_data))

    # warmup
    print(f"Warming up ({NUM_WARMUP} run)...", flush=True)
    for i in range(NUM_WARMUP):
        t0 = time.time()
        module.run()
        print(f"  Warmup {i+1}: {time.time()-t0:.1f}s", flush=True)

    # get output for sanity check
    output = module.get_output(0).numpy()
    top5 = np.argsort(output[0])[-5:][::-1]
    print(f"  output shape={output.shape}, argmax={output[0].argmax()}, "
          f"range=[{output.min():.3f}, {output.max():.3f}]", flush=True)

    # benchmark using TVM's benchmark API
    print(f"Benchmarking (number={BENCH_NUMBER}, repeat={BENCH_REPEAT})...", flush=True)
    timing = module.benchmark(dev, number=BENCH_NUMBER, repeat=BENCH_REPEAT)
    print(f"  {timing}", flush=True)

    mean_ms = float(timing.mean) * 1000
    std_ms = float(timing.std) * 1000
    median_ms = float(timing.median) * 1000
    print(f"\n  Results for {label}:")
    print(f"    Mean:   {mean_ms:.1f} ms")
    print(f"    Median: {median_ms:.1f} ms")
    print(f"    Std:    {std_ms:.1f} ms")
    print(f"    Top-5 class indices: {top5}")

    return {"label": label, "mean_ms": mean_ms, "std_ms": std_ms,
            "median_ms": median_ms}


def main():
    print("=" * 60)
    print("TVM ResNet50 Inference Benchmark on Raspberry Pi (QEMU)")
    print("=" * 60)
    print(f"Platform: {os.uname()}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Warmup runs: {NUM_WARMUP}, Benchmark: number={BENCH_NUMBER}, repeat={BENCH_REPEAT}")

    results = []

    # float32 model
    try:
        r = load_and_benchmark("resnet50_fp32_arm", "ResNet50 FP32")
        results.append(r)
    except Exception as e:
        print(f"FP32 model failed: {e}")

    # int8 model
    try:
        r = load_and_benchmark("resnet50_int8_arm", "ResNet50 INT8")
        results.append(r)
    except Exception as e:
        print(f"INT8 model failed: {e}")

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:20s}  mean={r['mean_ms']:.1f}ms  "
              f"median={r['median_ms']:.1f}ms  std={r['std_ms']:.1f}ms")
    if len(results) == 2:
        speedup = results[0]["mean_ms"] / results[1]["mean_ms"]
        print(f"\n  INT8 vs FP32 speedup: {speedup:.2f}x")
        print(f"  ratio (int8/fp32): {1/speedup:.2f}x")


if __name__ == "__main__":
    main()
