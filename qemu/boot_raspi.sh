#!/bin/bash
# Boot Raspberry Pi 3B on QEMU with TVM inference benchmark
# Usage: ./boot_raspi.sh [interactive]
#   No args: runs auto_bench.sh as init (automatic benchmark)
#   "interactive": drops to a bash shell

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

KERNEL="boot_files/kernel8.img"
DTB="boot_files/bcm2710-rpi-3-b-plus.dtb"
DISK="raspios_qemu.img"
MODELS="models.img"

# Check files exist
for f in "$KERNEL" "$DTB" "$DISK" "$MODELS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found"
        exit 1
    fi
done

# Choose init program
if [ "$1" = "interactive" ]; then
    INIT="init=/bin/bash"
    echo "=== Interactive mode: you will get a root shell ==="
    echo "  - mount USB:  mount /dev/sda /mnt/models"
    echo "  - run bench:  /home/pi/run_inference /mnt/models"
else
    INIT="init=/home/pi/auto_bench.sh"
    echo "=== Automatic benchmark mode ==="
    echo "  The benchmark will run automatically and then shut down."
fi

echo ""
echo "Starting QEMU Raspberry Pi 3B emulation..."
echo "  Kernel: $KERNEL"
echo "  DTB:    $DTB"
echo "  Disk:   $DISK"
echo "  Models: $MODELS (USB storage)"
echo ""
echo "NOTE: Full system emulation is SLOW. Boot may take 5-15 minutes."
echo "      Each inference run may take several minutes."
echo "Press Ctrl+A then X to force-quit QEMU."
echo ""

qemu-system-aarch64 \
    -machine raspi3b \
    -cpu cortex-a53 \
    -m 1G \
    -kernel "$KERNEL" \
    -dtb "$DTB" \
    -drive file="$DISK",format=raw,if=sd \
    -device usb-storage,drive=usbdisk \
    -drive file="$MODELS",format=raw,if=none,id=usbdisk \
    -append "root=/dev/mmcblk0p2 rw rootwait console=ttyAMA0 loglevel=3 $INIT" \
    -serial stdio \
    -no-reboot \
    -display none
