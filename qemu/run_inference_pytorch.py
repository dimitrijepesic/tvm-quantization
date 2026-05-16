#!/usr/bin/env python3
# pytorch inference runner for raspberry pi (qemu)
# fallback when tvm runtime is not available
# benchmarks resnet50 in float32 and int8 (dynamic quantization)
import time
import os
import numpy as np

import torch
import torch.nn as nn

MODEL_DIR = "/home/pi/models"
INPUT_SHAPE = (1, 3, 224, 224)
NUM_WARMUP = 2
NUM_RUNS = 10


def benchmark_model(model, label, input_tensor):
    # benchmark a pytorch model
    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"{'='*60}")

    model.eval()

    # warmup
    print(f"Warming up ({NUM_WARMUP} runs)...", flush=True)
    with torch.no_grad():
        for i in range(NUM_WARMUP):
            t0 = time.time()
            _ = model(input_tensor)
            print(f"  Warmup {i+1}: {time.time()-t0:.3f}s", flush=True)

    # benchmark
    print(f"Benchmarking ({NUM_RUNS} runs)...", flush=True)
    times = []
    with torch.no_grad():
        for i in range(NUM_RUNS):
            t0 = time.time()
            output = model(input_tensor)
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s", flush=True)

    # get output
    probs = torch.softmax(output, dim=1)
    top5 = torch.topk(probs, 5).indices[0].tolist()

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000

    # model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"\n  Results for {label}:")
    print(f"    Parameters size: {param_size / 1024 / 1024:.1f} MB")
    print(f"    Mean:   {mean_ms:.1f} ms")
    print(f"    Std:    {std_ms:.1f} ms")
    print(f"    Min:    {min_ms:.1f} ms")
    print(f"    Max:    {max_ms:.1f} ms")
    print(f"    Top-5 class indices: {top5}")

    return {"label": label, "mean_ms": mean_ms, "std_ms": std_ms,
            "min_ms": min_ms, "max_ms": max_ms, "param_mb": param_size/1024/1024}


def main():
    print("=" * 60)
    print("PyTorch ResNet50 Inference Benchmark on Raspberry Pi (QEMU)")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {os.uname()}")
    print(f"Input shape: {INPUT_SHAPE}")
    print(f"Warmup runs: {NUM_WARMUP}, Benchmark runs: {NUM_RUNS}")
    print(f"Threads: {torch.get_num_threads()}")

    input_tensor = torch.randn(*INPUT_SHAPE)
    results = []

    # float32 model
    print("\nLoading ResNet50 FP32...", flush=True)
    try:
        import torchvision
        model_fp32 = torchvision.models.resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "resnet50_fp32.pth")
        if os.path.exists(weights_path):
            model_fp32.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print("  Loaded saved weights")
        else:
            print("  Using random weights (no saved weights found)")
        r = benchmark_model(model_fp32, "ResNet50 FP32 (PyTorch)", input_tensor)
        results.append(r)
    except Exception as e:
        print(f"  FP32 failed: {e}")
        print("  Trying to load from saved TorchScript...", flush=True)
        try:
            ts_path = os.path.join(MODEL_DIR, "resnet50_fp32.pt")
            if os.path.exists(ts_path):
                model_fp32 = torch.jit.load(ts_path, map_location="cpu")
                r = benchmark_model(model_fp32, "ResNet50 FP32 (TorchScript)", input_tensor)
                results.append(r)
        except Exception as e2:
            print(f"  TorchScript fallback also failed: {e2}")

    # int8 quantized model
    print("\nPreparing ResNet50 INT8 (dynamic quantization)...", flush=True)
    try:
        if 'model_fp32' in dir() and hasattr(model_fp32, 'fc'):
            model_int8 = torch.quantization.quantize_dynamic(
                model_fp32, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            r = benchmark_model(model_int8, "ResNet50 INT8 (PyTorch dynamic quant)", input_tensor)
            results.append(r)
        else:
            model_fresh = torchvision.models.resnet50(weights=None)
            model_int8 = torch.quantization.quantize_dynamic(
                model_fresh, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            r = benchmark_model(model_int8, "ResNet50 INT8 (PyTorch dynamic quant)", input_tensor)
            results.append(r)
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")

    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:45s}  mean={r['mean_ms']:.1f}ms  "
              f"min={r['min_ms']:.1f}ms  max={r['max_ms']:.1f}ms  "
              f"params={r['param_mb']:.1f}MB")
    if len(results) == 2:
        speedup = results[0]["mean_ms"] / results[1]["mean_ms"]
        print(f"\n  INT8 vs FP32 speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
