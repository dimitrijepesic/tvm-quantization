# TVM inference benchmark — to be run INSIDE the aarch64 Ubuntu QEMU guest.
# Loads the cross-compiled fp32 and int8 ResNet50 artifacts (.so/.json/.params)
# from /home/ubuntu/, executes a warmup run plus a TVM benchmark() call, and
# prints a summary used to populate logs/qemu_bench.log on the host.
#
# Inside the guest, copy the six artifact files first:
#   scp -P 2222 experiments/resnet50_{fp32,int8}_arm.{so,json,params} \
#       ubuntu@localhost:~
# Then SSH in and run:
#   ssh -p 2222 ubuntu@localhost
#   python3 run_bench.py 2>&1 | tee qemu_bench.log
import tvm
from tvm.contrib import graph_executor
import numpy as np
import time


def load_and_bench(prefix, label):
    print(f"\n=== {label} ===", flush=True)
    lib = tvm.runtime.load_module(f"/home/ubuntu/{prefix}.so")
    with open(f"/home/ubuntu/{prefix}.json", "r") as f:
        graph = f.read()
    with open(f"/home/ubuntu/{prefix}.params", "rb") as f:
        params = f.read()
    dev = tvm.cpu(0)
    m = graph_executor.create(graph, lib, dev)
    m.load_params(params)
    img = np.random.randn(1, 3, 224, 224).astype("float32")
    m.set_input("input0", tvm.nd.array(img))
    print(f"[{label}] warmup run...", flush=True)
    t0 = time.time()
    m.run()
    print(f"[{label}] warmup: {time.time()-t0:.1f}s", flush=True)
    out = m.get_output(0).numpy()
    print(f"[{label}] output shape={out.shape}, argmax={out.argmax()}, "
          f"range=[{out.min():.3f}, {out.max():.3f}]", flush=True)
    print(f"[{label}] benchmarking (number=2, repeat=3)...", flush=True)
    timing = m.benchmark(dev, number=2, repeat=3)
    print(f"[{label}] {timing}", flush=True)
    return timing


t_fp32 = load_and_bench("resnet50_fp32_arm", "fp32")
t_int8 = load_and_bench("resnet50_int8_arm", "int8")

print("\n=== summary ===")
print(f"fp32: mean={t_fp32.mean*1000:.1f} ms")
print(f"int8: mean={t_int8.mean*1000:.1f} ms")
print(f"ratio (int8/fp32): {t_int8.mean/t_fp32.mean:.2f}x")
