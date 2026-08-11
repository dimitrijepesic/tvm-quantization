#!/bin/bash
# Boot an aarch64 Ubuntu 22.04 guest under QEMU's generic ARM `virt` machine
# with an emulated Cortex-A72 CPU -- matching the LLVM compile target
# (-mcpu=cortex-a72, see experiments/03_cross_compile_arm.py).
#
# Why not raspi3b? QEMU 6.2 (Ubuntu 22.04 default) only exposes raspi3b at the
# top of the Pi family; raspi3b emulates Cortex-A53, not the A72 we cross-
# compile for. raspi4b was only added in QEMU 8.2+. Using the virt machine
# with an explicit -cpu cortex-a72 gives us a faithful A72 execution of the
# same binary that would run on real Pi 4 hardware.
#
# Required files in the current directory:
#   ubuntu-22.04-server-cloudimg-arm64.img  -- the guest disk
#   seed.img                                -- cloud-init seed (creates
#                                              ubuntu user, ssh keys, etc.)
#   /usr/share/qemu-efi-aarch64/QEMU_EFI.fd -- UEFI firmware (apt: qemu-efi-aarch64)
#
# Networking: host:2222 -> guest:22 so SSH works without bridge configuration.
# Inside the guest, $HOME is /home/ubuntu; copy artifacts and run benchmarks
# from there (see qemu/run_bench_in_vm.py).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DISK="ubuntu-22.04-server-cloudimg-arm64.img"
SEED="seed.img"
EFI="/usr/share/qemu-efi-aarch64/QEMU_EFI.fd"

for f in "$DISK" "$SEED" "$EFI"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

echo "Booting Ubuntu aarch64 VM under QEMU virt + cortex-a72..."
echo "  SSH: ssh -p 2222 ubuntu@localhost (password: ubuntu)"
echo "  Press Ctrl-A then X to terminate QEMU."
echo ""

qemu-system-aarch64 \
    -M virt \
    -cpu cortex-a72 \
    -smp 4 \
    -m 4G \
    -bios "$EFI" \
    -nographic \
    -drive if=virtio,file="$DISK",format=qcow2 \
    -drive if=virtio,file="$SEED",format=raw \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -device virtio-net-pci,netdev=net0
