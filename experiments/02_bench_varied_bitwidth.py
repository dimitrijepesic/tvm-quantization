"""
Sub-byte quantization of ResNet50 in "logical" mode: nbit=4/2/1 but
dtype stays int8.

The quantization pass clips activations and weights to an N-bit effective
range, but the IR's storage dtype remains int8 (each value still occupies
a full byte — no actual packing). This isolates the question "does
sub-byte fail because of quantization logic or because of downstream
dtype handling?" — see Section 3.2.4 in the report.

fp32 and int8 baselines live in 01_bench_fp32_vs_int8.py — not repeated
here. Native sub-byte storage (dtype="int{n}") fails or hangs during
quantize/build; those cases live in diagnostics/probe_int{n}.py with
proper instrumentation.
"""
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import quantize

import numpy as np
import torch
import torchvision

# model import
# input - torchvision model with pretrained weights
# output - objects 'mod' and 'params' - graph in Relay IR and the trained weights

model_name = "resnet50"
model = getattr(torchvision.models, model_name)(weights = "IMAGENET1K_V2")
model = model.eval()

input_shape = [1,3,224,224]
input_data = torch.randn(input_shape) # dummy data
scripted_model = torch.jit.trace(model,input_data).eval()

img = np.random.randn(1,3,224,224).astype("float32")

# conversion of TorchScript graph to Relay IR
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# target and device
target = tvm.target.Target("llvm", host = "llvm")
dev = tvm.cpu(0)


# logical sub-byte configs — nbit < 8 with dtype kept at int8
# (storage = int8 → reuses default int8 kernels; native dtype="int{n}" fails: see diagnostics/)
configs = [
    ("int4_logical", 4),
    ("int2_logical", 2),
    ("int1_logical", 1),
]

for label, nbit in configs:
    print(f"\n=== {label} (nbit={nbit}, dtype=int8 storage) ===", flush=True)
    with quantize.qconfig(
        nbit_input = nbit,
        nbit_weight = nbit,
        nbit_activation = 32,
        dtype_input = "int8",
        dtype_weight = "int8",
        dtype_activation = "int32",
        calibrate_mode = "global_scale",
        global_scale = 8.0,
        skip_dense_layer = True,
        skip_conv_layers = [0],
    ):
        modq = quantize.quantize(mod, params = params)
    print(f"[{label}] quantize OK", flush=True)

    with tvm.transform.PassContext(opt_level = 3):
        libq = relay.build(modq, target = target, params = params)
    print(f"[{label}] build OK", flush=True)

    m = graph_executor.GraphModule(libq["default"](dev))
    m.set_input(input_name, tvm.nd.array(img.astype("float32")))
    print(f"{label}:", m.benchmark(dev, number=100), flush=True)
