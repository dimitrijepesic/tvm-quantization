# probe uint8 input / int8 weight quantization with dispatch logging
# tests whether the vnni-preferred dtype pairing (uint8 data, int8 weights)
# changes kernel dispatch or performance vs signed int8/int8
# run with: python3 probe_uint8.py 2>&1 | tee uint8_bench.log

import sys
import os
import atexit
import signal
import faulthandler
import logging

faulthandler.enable()

# enable te_compiler logging — prints which kernel each op dispatches to
# for conv2d we want to see "conv2d_NCHWc_int8.x86" (vnni path) vs
# "conv2d_NCHWc.x86" (generic fallback)
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("te_compiler").setLevel(logging.INFO)
logging.getLogger("compile_engine").setLevel(logging.INFO)

@atexit.register
def goodbye():
    print(">>> ATEXIT: shutting down normally <<<", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

def signal_handler(signum, frame):
    print(f">>> SIGNAL CAUGHT: {signum} ({signal.Signals(signum).name}) <<<",
          flush=True)
    sys.stdout.flush()
    sys.exit(128 + signum)

for sig in [signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGILL]:
    try:
        signal.signal(sig, signal_handler)
    except Exception as e:
        print(f"  Could not install handler for {sig}: {e}", flush=True)

print(">>> uint8 input / int8 weight quantization probe <<<", flush=True)
print(f">>> PID: {os.getpid()} <<<", flush=True)
print("", flush=True)

import tvm
from tvm import relay
import numpy as np
import torch
import torchvision
from tvm.contrib import graph_executor
from tvm.relay import quantize


def quantize_build_bench(label, dtype_input, dtype_weight, mod, params,
                          target, dev, input_name, img):
    print(f"\n{'='*60}", flush=True)
    print(f"=== {label}: dtype_input={dtype_input}, dtype_weight={dtype_weight}",
          flush=True)
    print(f"{'='*60}", flush=True)

    # quantize
    print(f"[{label}.1] quantize.quantize()...", flush=True)
    try:
        with quantize.qconfig(
            nbit_input=8, nbit_weight=8, nbit_activation=32,
            dtype_input=dtype_input,
            dtype_weight=dtype_weight,
            dtype_activation="int32",
            calibrate_mode="global_scale",
            global_scale=8.0,
            skip_dense_layer=True,
            skip_conv_layers=[0],
        ):
            modq = quantize.quantize(mod, params=params)
        print(f"[{label}.1] quantize OK", flush=True)
    except Exception as e:
        print(f"[{label}.1] QUANTIZE FAILED: {type(e).__name__}: {e}",
              flush=True)
        import traceback
        traceback.print_exc()
        return None

    # peek at ir to check actual dtypes
    ir_str = str(modq)
    print(f"[{label}.1]   IR length: {len(ir_str)} chars", flush=True)
    print(f"[{label}.1]   '{dtype_input}' substring count: "
          f"{ir_str.count(dtype_input)}", flush=True)
    if dtype_input != dtype_weight:
        print(f"[{label}.1]   '{dtype_weight}' substring count: "
              f"{ir_str.count(dtype_weight)}", flush=True)

    # build
    print(f"[{label}.2] relay.build()... (watch for 'Using ... for "
          f"nn.contrib_conv2d_NCHWc' lines below)", flush=True)
    sys.stdout.flush()
    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(modq, target=target, params=params)
        print(f"[{label}.2] build OK", flush=True)
    except Exception as e:
        print(f"[{label}.2] BUILD FAILED: {type(e).__name__}: {e}",
              flush=True)
        import traceback
        traceback.print_exc()
        return None

    # run + sanity check
    print(f"[{label}.3] running once for sanity check...", flush=True)
    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(input_name, tvm.nd.array(img.astype("float32")))
    m.run()
    out = m.get_output(0).numpy()
    print(f"[{label}.3]   argmax={out.argmax()}, "
          f"range=[{out.min():.3f}, {out.max():.3f}], "
          f"unique={len(np.unique(out))}", flush=True)
    if len(np.unique(out)) < 10:
        print(f"[{label}.3]   WARNING: very few unique values, "
              f"output may be degenerate", flush=True)

    # benchmark
    print(f"[{label}.4] benchmarking (number=100)...", flush=True)
    timing = m.benchmark(dev, number=100)
    print(f"[{label}.4] {timing}", flush=True)

    return {
        "label": label,
        "argmax": int(out.argmax()),
        "out_min": float(out.min()),
        "out_max": float(out.max()),
        "out_unique": int(len(np.unique(out))),
        "mean_ms": float(timing.mean) * 1000,
        "std_ms": float(timing.std) * 1000,
    }


# model setup
print("[setup] loading ResNet50...", flush=True)
model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
input_shape = [1, 3, 224, 224]
scripted = torch.jit.trace(model, torch.randn(input_shape)).eval()

img = np.random.randn(1, 3, 224, 224).astype("float32")
input_name = "input0"
shape_list = [(input_name, img.shape)]

print("[setup] importing to Relay...", flush=True)
mod, params = relay.frontend.from_pytorch(scripted, shape_list)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)

print(f"[setup] target = {target}", flush=True)

# experiments
results = []

# baseline: signed int8 / int8
r = quantize_build_bench("int8_signed", "int8", "int8",
                         mod, params, target, dev, input_name, img)
if r: results.append(r)

# the experiment: uint8 input, int8 weight (vnni dispatch requirement)
r = quantize_build_bench("uint8_input", "uint8", "int8",
                         mod, params, target, dev, input_name, img)
if r: results.append(r)

# summary
print("\n" + "="*60, flush=True)
print("SUMMARY", flush=True)
print("="*60, flush=True)
print(f"{'config':<20} {'mean (ms)':>12} {'argmax':>8} {'unique':>8} "
      f"{'range':>22}", flush=True)
for r in results:
    rng = f"[{r['out_min']:.2f}, {r['out_max']:.2f}]"
    print(f"{r['label']:<20} {r['mean_ms']:>12.2f} {r['argmax']:>8} "
          f"{r['out_unique']:>8} {rng:>22}", flush=True)

print("\n>>> SCRIPT REACHED END NORMALLY <<<", flush=True)
