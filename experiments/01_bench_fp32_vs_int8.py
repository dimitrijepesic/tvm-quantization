## Benchmark FP32 vs INT8 (quantized) ResNet50 inference on CPU using TVM

# imports
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

# benchmark for fp32
target = tvm.target.Target("llvm", host = "llvm")
dev = tvm.cpu(0)

with tvm.transform.PassContext(opt_level = 3):
    libfp32 = relay.build(mod, target=target, params=params)

mfp32 = graph_executor.GraphModule(libfp32["default"](dev))
mfp32.set_input(input_name, tvm.nd.array(img.astype("float32")))
timingfp32 = mfp32.benchmark(dev, number=100)
print(timingfp32)

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
    modint8 = quantize.quantize(mod,params = params)

# build the model
with tvm.transform.PassContext(opt_level = 3):
    libint8 = relay.build(modint8, target = target, params = params)

# create an executor from the Relay IR
mint8 = graph_executor.GraphModule(libint8["default"](dev))
mint8.set_input(input_name, tvm.nd.array(img.astype("float32")))
timingint8 = mint8.benchmark(dev, number=100)
print(timingint8)