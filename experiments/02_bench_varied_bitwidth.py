## benchmark varied bitwidth quantization of resnet50 inference on cpu using tvm
## try int8, int4, int2, int1 and note which precisions work
## for each bitwidth I try two modes:
##   - native: dtype_input/dtype_weight = "int{n}" (uses tvm's native sub-byte types)
##   - logical: dtype_input/dtype_weight = "int8" but nbit = n (quantizes to n bits, packed in int8 container)
## this reveals whether failures come from native dtype support or from the quantization logic itself
##
## known failure modes documented in diagnostics/failure_analysis.txt

# imports
import sys
import subprocess
import json
import traceback

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import quantize

import numpy as np
import torch
import torchvision


# quantize, build, and benchmark a model under a given config
# returns a dict capturing what happened at each stage
def benchmark_quantized(
    mod,
    params,
    target,
    dev,
    input_name,
    img,
    label,
    nbit_input=8,
    nbit_weight=8,
    dtype_input="int8",
    dtype_weight="int8",
    nbit_activation=32,
    dtype_activation="int32",
    calibrate_mode="global_scale",
    global_scale=8.0,
):
    result = {
        "label": label,
        "config": {
            "nbit_input": nbit_input,
            "nbit_weight": nbit_weight,
            "dtype_input": dtype_input,
            "dtype_weight": dtype_weight,
        },
        "quantize_ok": False,
        "build_ok": False,
        "bench_ok": False,
        "latency_mean_ms": None,
        "latency_std_ms": None,
        "error_stage": None,
        "error_msg": None,
    }

    # stage 1: quantize
    try:
        with quantize.qconfig(
            nbit_input=nbit_input,
            nbit_weight=nbit_weight,
            nbit_activation=nbit_activation,
            dtype_input=dtype_input,
            dtype_weight=dtype_weight,
            dtype_activation=dtype_activation,
            calibrate_mode=calibrate_mode,
            global_scale=global_scale,
            skip_dense_layer=True,
            skip_conv_layers=[0],
        ):
            modd = quantize.quantize(mod, params=params)
        result["quantize_ok"] = True
    except Exception as e:
        result["error_stage"] = "quantize"
        result["error_msg"] = str(e)
        return result

    # stage 2: build
    try:
        with tvm.transform.PassContext(opt_level=3):
            libb = relay.build(modd, target=target, params=params)
        result["build_ok"] = True
    except Exception as e:
        result["error_stage"] = "build"
        result["error_msg"] = str(e)
        return result

    # stage 3: benchmark
    try:
        m = graph_executor.GraphModule(libb["default"](dev))
        m.set_input(input_name, tvm.nd.array(img.astype("float32")))

        m.run()
        out = m.get_output(0).numpy()
        result["output_shape"] = list(out.shape)
        result["output_min"] = float(out.min()) if not np.isnan(out).any() else float('nan')
        result["output_max"] = float(out.max()) if not np.isnan(out).any() else float('nan')
        result["output_unique"] = int(len(np.unique(out)))
        result["output_argmax"] = int(out.argmax())
        result["output_nonzero_frac"] = float((out != 0).mean())

        timing = m.benchmark(dev, number=100)
        result["bench_ok"] = True
        result["latency_mean_ms"] = float(timing.mean) * 1000
        result["latency_std_ms"] = float(timing.std) * 1000
    except Exception as e:
        result["error_stage"] = "benchmark"
        result["error_msg"] = str(e)
        return result

    return result


def benchmark_float32(mod, params, target, dev, input_name, img, label="float32"):
    result = {
        "label": label,
        "config": {"dtype": "float32"},
        "quantize_ok": True,
        "build_ok": False,
        "bench_ok": False,
        "latency_mean_ms": None,
        "latency_std_ms": None,
        "error_stage": None,
        "error_msg": None,
    }

    try:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        result["build_ok"] = True
    except Exception as e:
        result["error_stage"] = "build"
        result["error_msg"] = str(e)
        return result

    try:
        m = graph_executor.GraphModule(lib["default"](dev))
        m.set_input(input_name, tvm.nd.array(img.astype("float32")))

        m.run()
        out = m.get_output(0).numpy()
        result["output_shape"] = list(out.shape)
        result["output_min"] = float(out.min())
        result["output_max"] = float(out.max())
        result["output_unique"] = int(len(np.unique(out)))
        result["output_argmax"] = int(out.argmax())
        result["output_nonzero_frac"] = float((out != 0).mean())

        timing = m.benchmark(dev, number=100)
        result["bench_ok"] = True
        result["latency_mean_ms"] = float(timing.mean) * 1000
        result["latency_std_ms"] = float(timing.std) * 1000
    except Exception as e:
        result["error_stage"] = "benchmark"
        result["error_msg"] = str(e)

    return result


def print_result(r):
    if r["bench_ok"]:
        print(
            f"  {r['label']:<18} {r['latency_mean_ms']:>8.2f} ms "
            f"\u00b1 {r['latency_std_ms']:>5.2f}",
            flush=True,
        )
        if r.get("output_argmax") is not None:
            print(
                f"    output: argmax={r['output_argmax']:>4d}, "
                f"unique={r['output_unique']:>4d}, "
                f"nonzero={r['output_nonzero_frac']:>5.1%}, "
                f"range=[{r['output_min']:>7.3f}, {r['output_max']:>7.3f}]",
                flush=True,
            )
    else:
        msg = r["error_msg"] or ""
        print(
            f"  {r['label']:<18} FAILED at {r['error_stage']}: {msg[:150]}",
            flush=True,
        )


# run a single quantize+build+bench attempt in a child process
# used for configs that segfault or hang — the child can die
# without killing the main experiment runner
def run_in_subprocess(label, nbit, dtype_input, dtype_weight, timeout=120):
    child_code = f"""
import json, sys, numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import quantize
import torch, torchvision

model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
scripted = torch.jit.trace(model, torch.randn([1,3,224,224])).eval()
img = np.random.randn(1,3,224,224).astype("float32")
mod, params = relay.frontend.from_pytorch(scripted, [("input0", (1,3,224,224))])
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)

result = {{"quantize_ok": False, "build_ok": False, "bench_ok": False,
           "error_stage": None, "error_msg": None,
           "latency_mean_ms": None, "latency_std_ms": None}}

try:
    with quantize.qconfig(
        nbit_input={nbit}, nbit_weight={nbit}, nbit_activation=32,
        dtype_input="{dtype_input}", dtype_weight="{dtype_weight}",
        dtype_activation="int32",
        calibrate_mode="global_scale", global_scale=8.0,
        skip_dense_layer=True, skip_conv_layers=[0],
    ):
        modd = quantize.quantize(mod, params=params)
    result["quantize_ok"] = True
except Exception as e:
    result["error_stage"] = "quantize"
    result["error_msg"] = str(e)
    print(json.dumps(result))
    sys.exit(0)

try:
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(modd, target=target, params=params)
    result["build_ok"] = True
except Exception as e:
    result["error_stage"] = "build"
    result["error_msg"] = str(e)
    print(json.dumps(result))
    sys.exit(0)

try:
    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input("input0", tvm.nd.array(img))
    timing = m.benchmark(dev, number=100)
    result["bench_ok"] = True
    result["latency_mean_ms"] = float(timing.mean) * 1000
    result["latency_std_ms"] = float(timing.std) * 1000
except Exception as e:
    result["error_stage"] = "benchmark"
    result["error_msg"] = str(e)

print(json.dumps(result))
"""
    result = {
        "label": label,
        "config": {"nbit_input": nbit, "nbit_weight": nbit,
                   "dtype_input": dtype_input, "dtype_weight": dtype_weight},
        "quantize_ok": False, "build_ok": False, "bench_ok": False,
        "latency_mean_ms": None, "latency_std_ms": None,
        "error_stage": None, "error_msg": None,
    }

    try:
        proc = subprocess.run(
            [sys.executable, "-c", child_code],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode < 0:
            # negative return code = killed by signal (e.g. -11 = SIGSEGV)
            import signal
            try:
                sig_name = signal.Signals(-proc.returncode).name
            except (ValueError, AttributeError):
                sig_name = str(-proc.returncode)
            result["error_stage"] = "quantize"
            result["error_msg"] = f"child killed by signal {sig_name} (exit code {proc.returncode})"
            return result
        # try to parse the json result from stdout
        # the child prints JSON as its last line; earlier lines may contain
        # TVM warnings (e.g. "operators have not been tuned")
        stdout = proc.stdout.strip()
        if stdout:
            for line in reversed(stdout.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        child_result = json.loads(line)
                        result.update(child_result)
                        result["label"] = label
                        break
                    except json.JSONDecodeError:
                        continue
    except subprocess.TimeoutExpired:
        result["error_stage"] = "quantize"
        result["error_msg"] = f"subprocess timed out after {timeout}s (likely infinite loop)"
    except Exception as e:
        result["error_stage"] = "subprocess"
        result["error_msg"] = str(e)

    return result


def print_summary_table(results):
    print("\n" + "=" * 70)
    print(f"{'Config':<18} {'Status':<12} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("=" * 70)
    for r in results:
        if r["bench_ok"]:
            status = "ok"
            mean = f"{r['latency_mean_ms']:.2f}"
            std = f"{r['latency_std_ms']:.2f}"
        else:
            status = f"fail@{r['error_stage']}"
            mean = "-"
            std = "-"
        print(f"{r['label']:<18} {status:<12} {mean:>12} {std:>12}")
    print("=" * 70)


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

# experiment matrix
# "logical" configs: sub-byte nbit but int8 storage (reuses int8 kernels)
# "native" configs: sub-byte storage dtype (likely fails in build or earlier)
quantized_configs = [
    # int8 baseline
    {"label": "int8",           "nbit_input": 8, "nbit_weight": 8, "dtype_input": "int8",  "dtype_weight": "int8"},
    # int4
    {"label": "int4_logical",   "nbit_input": 4, "nbit_weight": 4, "dtype_input": "int8",  "dtype_weight": "int8"},
    {"label": "int4_native",    "nbit_input": 4, "nbit_weight": 4, "dtype_input": "int4",  "dtype_weight": "int4"},
    # int2
    {"label": "int2_logical",   "nbit_input": 2, "nbit_weight": 2, "dtype_input": "int8",  "dtype_weight": "int8"},
    {"label": "int2_native",    "nbit_input": 2, "nbit_weight": 2, "dtype_input": "int2",  "dtype_weight": "int2"},
    # int1
    {"label": "int1_logical",   "nbit_input": 1, "nbit_weight": 1, "dtype_input": "int8",  "dtype_weight": "int8"},
    {"label": "int1_native",    "nbit_input": 1, "nbit_weight": 1, "dtype_input": "int1",  "dtype_weight": "int1"},
]

# run experiments

results = []

print("Running float32 baseline...", flush=True)
r = benchmark_float32(mod, params, target, dev, input_name, img)
results.append(r)
print_result(r)

# int2_native segfaults during quantize.quantize() — not catchable via try/except
# run it in a subprocess so it cannot kill the main process
SUBPROCESS_CONFIGS = {"int2_native"}

for cfg in quantized_configs:
    print(f"\n>>> Starting {cfg['label']}", flush=True)
    if cfg["label"] in SUBPROCESS_CONFIGS:
        print(f"  (running in subprocess — known to segfault)", flush=True)
        r = run_in_subprocess(
            cfg["label"], cfg["nbit_input"],
            cfg["dtype_input"], cfg["dtype_weight"],
            timeout=120,
        )
    else:
        r = benchmark_quantized(mod, params, target, dev, input_name, img, **cfg)
    results.append(r)
    print_result(r)

# summary

print_summary_table(results)
print("\nDone.")
