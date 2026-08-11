# probe native int2 quantization. print python stack every 30s during hang
# run with: python3 probe_int2.py 2>&1 | tee ../logs/int2_native.log
import sys
import os
import atexit
import signal
import faulthandler
import threading

faulthandler.enable()

# dump traces every 30 seconds while the process runs
# even when tvm is stuck in a tight loop, this alarm-based dumper
# will fire and show what python frame the c++ extension was called from
faulthandler.dump_traceback_later(timeout=30, repeat=True, file=sys.stderr)

@atexit.register
def goodbye():
    print(">>> ATEXIT: shutting down <<<", flush=True)

print(">>> Native int2 quantization probe (with periodic stack dumps) <<<",
      flush=True)
print(f">>> PID: {os.getpid()} <<<", flush=True)
print(">>> A Python stack trace will be dumped every 30 seconds <<<", flush=True)
print(">>> Look for repeated frames — that's where the loop is <<<", flush=True)
print("", flush=True)

import tvm
from tvm import relay
import numpy as np
import torch
import torchvision
from tvm.relay import quantize

print("[1] Loading ResNet50...", flush=True)
model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
input_data = torch.randn([1, 3, 224, 224])
scripted = torch.jit.trace(model, input_data).eval()

img = np.random.randn(1, 3, 224, 224).astype("float32")
input_name = "input0"
shape_list = [(input_name, img.shape)]

print("[2] Converting to Relay...", flush=True)
mod, params = relay.frontend.from_pytorch(scripted, shape_list)

target = tvm.target.Target("llvm", host="llvm")

print("[3] Entering quantize.qconfig context with nbit=2, dtype='int2'...",
      flush=True)
sys.stdout.flush()

try:
    with quantize.qconfig(
        nbit_input=2,
        nbit_weight=2,
        nbit_activation=32,
        dtype_input="int2",
        dtype_weight="int2",
        dtype_activation="int32",
        calibrate_mode="global_scale",
        global_scale=8.0,
        skip_dense_layer=True,
        skip_conv_layers=[0],
    ):
        print("[3.1] Calling quantize.quantize() — this is where it hangs",
              flush=True)
        sys.stdout.flush()
        modd = quantize.quantize(mod, params=params)
        print("[3.2] quantize.quantize() returned (unexpected!)", flush=True)
except KeyboardInterrupt:
    print("\n[3.X] KeyboardInterrupt caught", flush=True)
    print("[3.X] The faulthandler dumps above show where the loop is",
          flush=True)
    sys.exit(130)
except Exception as e:
    print(f"[3.X] EXCEPTION: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[3] Quantize succeeded — UNEXPECTED", flush=True)
