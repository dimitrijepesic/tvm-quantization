"""
Cross-compile fp32 and int8 ResNet50 to aarch64 (Cortex-A72, Raspberry Pi 4).

Workflow:
  1. Load and import ResNet50 as in 01_bench_fp32_vs_int8.py
  2. Set target to llvm with arm_cpu device, aarch64-linux-gnu triple,
     cortex-a72 CPU
  3. Build fp32 model, export .so/.json/.params via cross-compiler
     (aarch64-linux-gnu-gcc)
  4. Apply int8 quantization and repeat the export for the quantized
     model

Output: six files (resnet50_{fp32,int8}_arm.{so,json,params}) in the
experiments/ directory. These are loaded inside the aarch64 QEMU guest
by qemu/run_bench_in_vm.py.

Run: python 03_cross_compile_arm.py 2>&1 | tee ../logs/cross_compile_arm.log
"""
import os

import tvm
from tvm import relay
from tvm.contrib import cc as _cc
from tvm.relay import quantize

import numpy as np
import torch
import torchvision

# cross-compilation helper
# uses aarch64-linux-gnu-gcc to produce an arm shared library from tvm's generated code
def cross_cc(output, files, options=None, cc="aarch64-linux-gnu-gcc"):
    return _cc.create_shared(output, files, options, cc=cc)

# model import
# input - torchvision model with pretrained weights
# output - objects 'mod' and 'params' - graph in relay ir and the trained weights

model_name = "resnet50"
model = getattr(torchvision.models, model_name)(weights = "IMAGENET1K_V2")
model = model.eval()

input_shape = [1,3,224,224]
input_data = torch.randn(input_shape) # dummy data
scripted_model = torch.jit.trace(model,input_data).eval()

img = np.random.randn(1,3,224,224).astype("float32")

# conversion of torchscript graph to relay ir
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# target: raspberry pi 3/4 (cortex-a72, aarch64)
target = tvm.target.Target(
    "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=cortex-a72"
)

# output directory
out_dir = os.path.dirname(os.path.abspath(__file__))

# cross-compile fp32 model
with tvm.transform.PassContext(opt_level = 3):
    libfp32 = relay.build(mod, target = target, params = params)

libfp32.export_library(os.path.join(out_dir, "resnet50_fp32_arm.so"), fcompile = cross_cc)
with open(os.path.join(out_dir, "resnet50_fp32_arm.json"), "w") as f:
    f.write(libfp32.get_graph_json())
with open(os.path.join(out_dir, "resnet50_fp32_arm.params"), "wb") as f:
    f.write(tvm.runtime.save_param_dict(libfp32.get_params()))
print("exported fp32 ARM artifacts", flush=True)

# quantization to int8
with quantize.qconfig(
    nbit_input = 8,
    nbit_weight = 8,
    nbit_activation = 32,
    dtype_input = "int8",
    dtype_weight = "int8",
    dtype_activation = "int32",
    calibrate_mode = "global_scale",
    global_scale = 8.0,
    skip_dense_layer = True,
    skip_conv_layers = [0],
):
    modint8 = quantize.quantize(mod, params = params)

# cross-compile int8 model
with tvm.transform.PassContext(opt_level = 3):
    libint8 = relay.build(modint8, target = target, params = params)

libint8.export_library(os.path.join(out_dir, "resnet50_int8_arm.so"), fcompile = cross_cc)
with open(os.path.join(out_dir, "resnet50_int8_arm.json"), "w") as f:
    f.write(libint8.get_graph_json())
with open(os.path.join(out_dir, "resnet50_int8_arm.params"), "wb") as f:
    f.write(tvm.runtime.save_param_dict(libint8.get_params()))
print("exported int8 ARM artifacts", flush=True)

print("done")
