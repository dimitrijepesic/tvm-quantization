# QEMU aarch64 Ubuntu emulation (Cortex-A72)

## Setup rationale

The project specification asks for a Raspberry Pi emulator under QEMU. The
QEMU version available on our build host (`qemu-system-aarch64` 6.2.0) only
ships Raspberry Pi machine types up to `raspi3b` (Cortex-A53); the
`raspi4b` model (Cortex-A72) was added in QEMU 8.2+ and is not available
here. Since our cross-compile target is `-mcpu=cortex-a72` (the Pi 4's
CPU; see `experiments/03_cross_compile_arm.py`), running on `raspi3b`
would force a Cortex-A53 microarchitecture, mismatched with the LLVM
scheduler. The generic ARM `virt` machine accepts an explicit `-cpu`
flag, so we use `virt` with `-cpu cortex-a72` as the closest faithful
emulation of a Raspberry Pi 4 environment.

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

## Boot

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

## Run the benchmark

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

## Notes

- QEMU's TCG software CPU emulator does not preserve A72's pipeline depth,
  branch predictor, or cache hierarchy; absolute latencies are much
  higher than native Pi 4. The fp32-vs-int8 ratio remains informative
  because both configurations pay the same emulation cost.
- For a PyTorch fallback (if the TVM runtime won't build inside the
  guest), see `run_inference_pytorch.py`.
