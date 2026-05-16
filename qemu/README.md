# QEMU Raspberry Pi Emulation Setup

## Prerequisites

```bash
apt install qemu-system-aarch64
```

## Required Files

The boot script expects the following files in `qemu/`:

1. **boot_files/kernel8.img** — aarch64 kernel from a Raspberry Pi OS image
2. **boot_files/bcm2710-rpi-3-b-plus.dtb** — device tree blob
3. **raspios_qemu.img** — Raspberry Pi OS root filesystem (raw disk image)
4. **models.img** — raw disk image containing the cross-compiled TVM models

## Obtaining Boot Files

Extract kernel and DTB from a Raspberry Pi OS image:

```bash
# Download Raspberry Pi OS Lite (64-bit) from:
# https://www.raspberrypi.com/software/operating-systems/

# Mount the boot partition (first partition of the .img):
losetup -fP raspios.img
mount /dev/loop0p1 /mnt/boot

# Copy kernel and DTB:
mkdir -p boot_files
cp /mnt/boot/kernel8.img boot_files/
cp /mnt/boot/bcm2710-rpi-3-b-plus.dtb boot_files/

# Or use extract_boot.py on a raw FAT32 boot partition image:
python3 extract_boot.py
```

## Creating the Models Image

```bash
# Create a disk image with the cross-compiled model files:
dd if=/dev/zero of=models.img bs=1M count=512
mkfs.ext4 models.img
mkdir /tmp/models_mnt && mount models.img /tmp/models_mnt
cp experiments/resnet50_*_arm.{so,json,params} /tmp/models_mnt/
umount /tmp/models_mnt
```

## Preparing the Root Filesystem

The root filesystem must have TVM runtime installed:

```bash
# Boot interactively first:
./boot_raspi.sh interactive

# Inside QEMU guest:
apt install python3 python3-numpy
# Build TVM runtime from source (runtime-only, no LLVM needed):
cd /home/pi/tvm && make runtime
```

## Running

```bash
# Automatic benchmark (runs and shuts down):
./boot_raspi.sh

# Interactive shell:
./boot_raspi.sh interactive
```

## Notes

- The `raspi3b` machine type emulates Cortex-A53 with 1GB RAM
- Full-system emulation is slow: boot takes 5-15 minutes, each inference several minutes
- The compiled models target Cortex-A72 (`-mcpu=cortex-a72`) but execute correctly under A53 emulation since both share the ARMv8-A NEON ISA
- Press Ctrl+A then X to force-quit QEMU
