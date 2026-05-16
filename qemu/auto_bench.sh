#!/bin/bash
# Auto-benchmark script - runs as init (PID 1) inside QEMU
# Mounts filesystems, finds models, runs TVM inference, then shuts down.

# Mount essential filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev

echo "============================================================"
echo "Raspberry Pi QEMU Inference Benchmark"
echo "============================================================"
echo "Kernel: $(uname -a)"
echo "CPU info:"
cat /proc/cpuinfo | head -20

# Remount root as read-write
mount -o remount,rw /

# Wait for devices to settle
sleep 3

# Try to find model files on USB storage
echo ""
echo "Looking for model files..."
mkdir -p /mnt/models
MODEL_DIR=""

for dev in /dev/sda /dev/sda1 /dev/sdb /dev/sdb1; do
    if [ -b "$dev" ]; then
        echo "  Trying $dev..."
        mount "$dev" /mnt/models 2>/dev/null
        if [ -f /mnt/models/resnet50_int8_arm.so ]; then
            echo "  Found models on $dev"
            MODEL_DIR=/mnt/models
            break
        fi
        umount /mnt/models 2>/dev/null
    fi
done

# Fallback: check if models are on root filesystem
if [ -z "$MODEL_DIR" ]; then
    if [ -f /home/pi/models/resnet50_int8_arm.so ]; then
        MODEL_DIR=/home/pi/models
        echo "  Found models in /home/pi/models"
    fi
fi

if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: No model files found!"
    echo "Dropping to shell..."
    exec /bin/bash
fi

echo "Model directory: $MODEL_DIR"
ls -lh "$MODEL_DIR/"

# Make runner executable
chmod +x /home/pi/run_inference

# Run the benchmark
echo ""
echo "Starting TVM inference benchmark..."
echo ""
/home/pi/run_inference "$MODEL_DIR" 2>&1

echo ""
echo "============================================================"
echo "Benchmark complete. Shutting down..."
echo "============================================================"

# Sync and power off
sync
poweroff -f
