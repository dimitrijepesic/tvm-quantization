# QEMU AArch64 emulation (Cortex-A72)

Two complementary QEMU setups are used, both emulating a Cortex-A72 (the
Raspberry Pi 4's CPU, matching the `-mcpu=cortex-a72` compile target in
`experiments/03_cross_compile_arm.py`):

1. **`virt` machine + Ubuntu guest** — full-system emulation with VirtIO
   storage and user-mode networking. Used for the main fp32-vs-int8
   ResNet50 benchmark (`logs/qemu_bench.log`), because networking makes
   it possible to install Python/NumPy and build the TVM runtime inside
   the guest.
2. **`raspi4b` machine + custom initramfs** — QEMU's actual Raspberry
   Pi 4 board model. Used to validate the `virt` results on the real
   Pi 4 machine model (`logs/raspi4b_*.log`). Requires a source-built
   QEMU >= 9.0 and has no usable networking, so the benchmarks run as
   self-contained PID 1 binaries instead of a full OS.

## Why two machines?

The project specification asks for a Raspberry Pi emulator under QEMU.
The distro QEMU on the build host (`qemu-system-aarch64` 6.2.0, Ubuntu
22.04) only ships Raspberry Pi machine types up to `raspi3b`
(Cortex-A53); the `raspi4b` model (Cortex-A72) was added in QEMU 9.0.
Running on `raspi3b` would force a Cortex-A53 microarchitecture,
mismatched with the LLVM compile target. The generic ARM `virt` machine
accepts an explicit `-cpu` flag, so the main benchmark uses `virt` with
`-cpu cortex-a72`.

For the `raspi4b` runs, QEMU 9.2.0 was built from source
(`./configure --target-list=aarch64-softmmu --enable-slirp`). The
`raspi4b` machine's device tree disables the bcm2711 peripherals QEMU
does not emulate — including `brcm,bcm2711-genet-v5`, the Pi 4's
Ethernet controller — so nothing can be installed inside that guest and
the TVM *Python* runtime cannot be provisioned there. The `raspi4b`
benchmarks are therefore statically-provisioned binaries run as PID 1
from an initramfs (see "raspi4b runs" below). Full details are in
Section 4.2 of the report.

## Prerequisites (host)

```bash
sudo apt install qemu-system-arm qemu-efi-aarch64 cloud-image-utils
```

Then download the Ubuntu 22.04 arm64 cloud image into this directory:

```bash
wget https://cloud-images.ubuntu.com/releases/22.04/release/ubuntu-22.04-server-cloudimg-arm64.img
qemu-img resize ubuntu-22.04-server-cloudimg-arm64.img +10G
```

## cloud-init seed image

`seed.img` is a tiny FAT image that cloud-init reads on first boot to set
the `ubuntu` user's password and authorize SSH. Generate it from a
`user-data` file containing:

```yaml
#cloud-config
password: ubuntu
chpasswd: { expire: False }
ssh_pwauth: True
```

and an empty `meta-data` file:

```bash
echo "instance-id: tvm-arm-vm" > meta-data
cloud-localds seed.img user-data meta-data
```

## Boot the `virt` VM

```bash
./boot_virt.sh
```

The script launches QEMU with:

- `-M virt` (generic ARM virt machine)
- `-cpu cortex-a72` (matches the LLVM compile target)
- `-smp 4 -m 4G` (4 vCPUs, 4 GB RAM)
- `-bios /usr/share/qemu-efi-aarch64/QEMU_EFI.fd` (UEFI boot)
- `hostfwd=tcp::2222-:22` (SSH from host:2222 -> guest:22)

First boot takes a few minutes for cloud-init to grow the rootfs and set
credentials. Subsequent boots are fast.

## TVM runtime inside the guest

SSH in and build the TVM runtime once:

```bash
ssh -p 2222 ubuntu@localhost   # password: ubuntu
sudo apt update && sudo apt install -y python3-pip python3-numpy cmake g++
git clone --recursive https://github.com/apache/tvm.git ~/tvm -b v0.15.0
cd ~/tvm/build
cmake -DUSE_LLVM=OFF ..    # runtime-only build (no LLVM needed)
make runtime -j$(nproc)
# Make the runtime importable
echo 'export PYTHONPATH=$HOME/tvm/python:$PYTHONPATH' >> ~/.bashrc
echo 'export TVM_LIBRARY_PATH=$HOME/tvm/build' >> ~/.bashrc
```

## Run the `virt` benchmark

From the host, copy the six artifact files into the VM:

```bash
scp -P 2222 \
    experiments/resnet50_fp32_arm.{so,json,params} \
    experiments/resnet50_int8_arm.{so,json,params} \
    ubuntu@localhost:~/
scp -P 2222 qemu/run_bench_in_vm.py ubuntu@localhost:~/
```

Then run the benchmark inside the VM and tee the log back to the host:

```bash
ssh -p 2222 ubuntu@localhost "python3 run_bench_in_vm.py" \
    | tee logs/qemu_bench.log
```

The output is the file in `logs/qemu_bench.log`.

## raspi4b runs (QEMU >= 9.0, built from source)

Two self-contained benchmarks run as PID 1 from a custom initramfs on
the `raspi4b` machine, booting the Raspberry Pi OS Bookworm arm64 kernel
(Linux 6.6.51) with `rdinit=/init`:

- **`microbench.c`** — statically linked scalar MAC-throughput loops
  (fp32/int32/int16/int8) plus a `/proc/cpuinfo` dump proving all four
  emulated cores are Cortex-A72 (MIDR `0x410fd083`). Log:
  `logs/raspi4b_microbench.log`. Build:

  ```bash
  aarch64-linux-gnu-gcc -O2 -mcpu=cortex-a72 -static -o microbench microbench.c
  ```

- **`resnet50_bench.cpp`** — links the TVM C++ runtime statically, loads
  a cross-compiled `resnet50_*_arm.{so,json,params}` triple, runs one
  warmup plus five timed inferences, prints results over `/dev/kmsg`,
  and powers the VM off. The full build command (requires an aarch64
  build of `libtvm_runtime.a`) is in the file's header comment. Logs:
  - `logs/raspi4b_resnet50_int8.log` — int8 completed end-to-end
    (mean 5.725 s over 5 runs, same output argmax as the `virt` run).
  - `logs/raspi4b_resnet50_fp32_oom.log` — fp32 could not run: the
    `raspi4b` machine's RAM is fixed at 2 GiB, the ~385 MB fp32
    artifact triple pushes the initramfs past what the kernel can
    unpack (`ENOSPC` mid-write), and the truncated `.so` faults with
    SIGBUS when mapped.

The compiled binaries are not checked in; both rebuild from the sources
in this directory with the commands above.

## Notes

- QEMU's TCG software CPU emulation does not preserve the A72's pipeline
  depth, branch predictor, or cache hierarchy; absolute latencies are
  much higher than native Pi 4. The fp32-vs-int8 *ratio* remains
  informative because both configurations pay the same emulation cost.
