## inspect the generated code inside a TVM compiled module
## demonstrates how to view TVM's generated code at different levels:
##   - LLVM IR (intermediate representation)
##   - assembly (target machine code)
## this answers the question: "how can you see the TVM code in the compiled module?"

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import quantize

import numpy as np
import torch
import torchvision

# model import
model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224)).eval()
mod, params = relay.frontend.from_pytorch(scripted, [("input0", (1, 3, 224, 224))])

target = tvm.target.Target("llvm", host="llvm")

# compile fp32 model
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# inspect the compiled module's generated code
print("=" * 70)
print("LLVM IR (first 3000 chars):")
print("=" * 70)
llvm_ir = lib.get_lib().get_source("ll")
print(llvm_ir[:3000])
print(f"\n... ({len(llvm_ir)} total chars)")

print("\n" + "=" * 70)
print("Assembly (first 3000 chars):")
print("=" * 70)
asm = lib.get_lib().get_source("asm")
print(asm[:3000])
print(f"\n... ({len(asm)} total chars)")

# show the graph JSON structure (operator names and execution order)
print("\n" + "=" * 70)
print("Graph JSON excerpt (first 2000 chars):")
print("=" * 70)
import json
graph = json.loads(lib.get_graph_json())
print(f"Number of nodes: {len(graph['nodes'])}")
print(f"Node op types: {set(n['op'] for n in graph['nodes'])}")
print("\nFirst 20 nodes:")
for i, node in enumerate(graph["nodes"][:20]):
    print(f"  [{i:3d}] op={node['op']:<12s} name={node.get('name', '')}")

print("\nDone.")
