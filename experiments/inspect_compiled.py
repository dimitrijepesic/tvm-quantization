"""
Inspect TVM's generated code for fp32 and int8 ResNet50 builds.

This auxiliary script demonstrates that compiled modules are not black
boxes — TVM exposes generated LLVM IR (lib.get_source("ll")) and
assembly (lib.get_source("asm")) through its API.

The key empirical check here is whether int8-specific x86 instructions
(VPDPBUSD, the VNNI fused dot-product) appear in the generated assembly
for the int8 model. The report (Section 3.1) argues that with the
default "llvm" target (no -mcpu advertising VNNI), dispatch falls
through to the generic kernel — so VPDPBUSD should NOT appear, even in
the int8 build. This script verifies that empirically.

Run: python inspect_compiled.py 2>&1 | tee ../logs/inspect_compiled.log
"""
import json
import re

import tvm
from tvm import relay
from tvm.relay import quantize

import torch
import torchvision


def build(label, mod, params, target):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    print(f"\n=== {label} ===", flush=True)
    return lib


def inspect(label, lib):
    print(f"\n--- inspecting {label} ---", flush=True)

    llvm_ir = lib.get_lib().get_source("ll")
    asm = lib.get_lib().get_source("asm")
    graph = json.loads(lib.get_graph_json())

    # LLVM IR summary
    funcs = re.findall(r"^define\s+\w+\s+@(\w+)", llvm_ir, re.MULTILINE)
    conv_funcs = [f for f in funcs if "conv" in f.lower()]
    vec_uses = len(re.findall(r"<\d+\s+x\s+(?:float|i\d+)>", llvm_ir))
    print(f"  LLVM IR: {len(llvm_ir)} chars, {len(llvm_ir.splitlines())} lines")
    print(f"  LLVM functions: {len(funcs)} total, "
          f"{len(conv_funcs)} conv-related")
    print(f"  Vector type usages (SIMD): {vec_uses}")

    # Assembly: look for specific instruction families
    avx_count = len(re.findall(
        r"\b(?:vmulps|vaddps|vfmadd\w+|vpmaddubsw)\b", asm))
    vnni_count = len(re.findall(r"\bvpdpbusd\b", asm))
    print(f"  Assembly: {len(asm)} chars, {len(asm.splitlines())} lines")
    print(f"  AVX/FMA vector instructions: {avx_count}")
    print(f"  VPDPBUSD (VNNI int8 fused dot-product): {vnni_count}")
    if vnni_count == 0:
        print(f"    -> VNNI NOT emitted: dispatch went to generic kernel")
    else:
        print(f"    -> VNNI emitted: dispatch hit conv2d_NCHWc_int8.x86")

    # Graph: operator distribution
    op_types = {}
    for node in graph["nodes"]:
        op_types[node["op"]] = op_types.get(node["op"], 0) + 1
    fused_pat = re.compile(r"tvmgen_default_fused_(.+)")
    fused_ops = {}
    for node in graph["nodes"]:
        if node["op"] == "tvm_op":
            m = fused_pat.match(node.get("name", ""))
            if m:
                key = m.group(1)
                fused_ops[key] = fused_ops.get(key, 0) + 1
    print(f"  Graph nodes: {len(graph['nodes'])} total, "
          f"op type distribution: {op_types}")
    top = sorted(fused_ops.items(), key=lambda kv: -kv[1])[:8]
    print(f"  Top fused operator patterns:")
    for name, n in top:
        print(f"    {name:60s} x {n}")


# model import
model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224)).eval()
mod, params = relay.frontend.from_pytorch(
    scripted, [("input0", (1, 3, 224, 224))])
target = tvm.target.Target("llvm", host="llvm")

# fp32
lib_fp32 = build("fp32", mod, params, target)
inspect("fp32", lib_fp32)

# int8
with quantize.qconfig(
    nbit_input=8, nbit_weight=8, nbit_activation=32,
    dtype_input="int8", dtype_weight="int8", dtype_activation="int32",
    calibrate_mode="global_scale", global_scale=8.0,
    skip_dense_layer=True, skip_conv_layers=[0],
):
    mod_int8 = quantize.quantize(mod, params=params)
lib_int8 = build("int8 (signed)", mod_int8, params, target)
inspect("int8 (signed)", lib_int8)

print("\nDone.", flush=True)
