# probe native int4 quantization with heavy diagnostic instrumentation
# run with: python3 probe_int4.py 2>&1 | tee ../logs/int4_native.log
import sys
import os
import atexit
import signal
import faulthandler
import re

faulthandler.enable()

# dump python+native stack every 30s while alive. crucial if relay.build hangs:
# repeated identical frames identify the loop site.
faulthandler.dump_traceback_later(timeout=30, repeat=True, file=sys.stderr)

@atexit.register
def goodbye():
    print(">>> ATEXIT: Python is shutting down normally <<<", flush=True)
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

print(">>> Native int4 quantization probe (instrumented) <<<", flush=True)
print(f">>> PID: {os.getpid()} <<<", flush=True)
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

print("[3] Quantizing with nbit=4, dtype='int4'...", flush=True)
with quantize.qconfig(
    nbit_input=4,
    nbit_weight=4,
    nbit_activation=32,
    dtype_input="int4",
    dtype_weight="int4",
    dtype_activation="int32",
    calibrate_mode="global_scale",
    global_scale=8.0,
    skip_dense_layer=True,
    skip_conv_layers=[0],
):
    modd = quantize.quantize(mod, params=params)
print("[3] Quantize succeeded.", flush=True)

print("[4] Inspecting quantized IR...", flush=True)
ir_str = str(modd)
print(f"[4]   IR length: {len(ir_str)} chars", flush=True)
print(f"[4]   First 500 chars of IR:", flush=True)
print(ir_str[:500], flush=True)
print(f"[4]   Searching IR for 'int4' references...", flush=True)
print(f"[4]   Count of 'int4' substring: {ir_str.count('int4')}", flush=True)

print("[4.5] Detailed IR dtype + operator inventory pre-build:", flush=True)
for dtype_token in ("int4", "int8", "int16", "int32", "uint4", "uint8",
                    "float16", "float32"):
    cnt = ir_str.count(dtype_token)
    if cnt:
        print(f"[4.5]   dtype '{dtype_token}': {cnt} mentions", flush=True)
op_calls = re.findall(r"\b(nn\.\w+|cast|clip|right_shift|multiply|add|"
                      r"annotation\.\w+|stop_fusion)\b", ir_str)
op_counts = {}
for op in op_calls:
    op_counts[op] = op_counts.get(op, 0) + 1
print(f"[4.5]   distinct ops: {len(op_counts)}; "
      f"top 12 by frequency:", flush=True)
for op, n in sorted(op_counts.items(), key=lambda kv: -kv[1])[:12]:
    print(f"[4.5]     {op:24s} x {n}", flush=True)
print(f"[4.5]   total IR text size: {len(ir_str)} chars "
      f"(~{len(ir_str.splitlines())} lines)", flush=True)

print("[5] BEFORE relay.build", flush=True)
sys.stdout.flush()
sys.stderr.flush()

try:
    with tvm.transform.PassContext(opt_level=3):
        print("[5.1] Inside PassContext, about to call relay.build", flush=True)
        sys.stdout.flush()
        lib = relay.build(modd, target=target, params=params)
        print("[5.2] relay.build returned!", flush=True)
        sys.stdout.flush()
    print("[6] AFTER PassContext exit", flush=True)
    sys.stdout.flush()
except Exception as e:
    print(f"[5.X] EXCEPTION caught: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[7] lib = {lib}", flush=True)
print(f"[7] lib type = {type(lib)}", flush=True)

print("[8] Creating GraphModule...", flush=True)
dev = tvm.cpu(0)
from tvm.contrib import graph_executor
m = graph_executor.GraphModule(lib["default"](dev))
print("[8] GraphModule created", flush=True)

print("[9] Setting input...", flush=True)
m.set_input("input0", tvm.nd.array(img))
print("[9] Input set", flush=True)

print("[10] Running...", flush=True)
m.run()
print("[10] Run completed", flush=True)

print("[11] Getting output...", flush=True)
out = m.get_output(0).numpy()
print(f"[11] output: shape={out.shape}, range=[{out.min():.3f}, {out.max():.3f}]",
      flush=True)

print(">>> SCRIPT REACHED END NORMALLY <<<", flush=True)
