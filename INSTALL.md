# Installation

## Python dependencies

```
pip install -r requirements.txt
```

## TVM (build from source, v0.15.0)

```
git clone --recursive --branch v0.15.0 https://github.com/apache/tvm.git
cd tvm && mkdir build && cp cmake/config.cmake build/
# Edit build/config.cmake: set USE_LLVM to ON
cd build && cmake .. && make -j$(nproc)
cd ../python && pip install -e .
```

## Cross-compilation toolchain (for experiments/03_cross_compile_arm.py)

```
sudo apt install gcc-aarch64-linux-gnu
```

## QEMU emulation (for qemu/ benchmarks)

```
sudo apt install qemu-system-arm qemu-efi-aarch64 cloud-image-utils
```

This is sufficient for the `virt`-machine benchmark. The `raspi4b`
machine runs additionally require QEMU >= 9.0 built from source; see
`qemu/README.md` for details on both setups.
