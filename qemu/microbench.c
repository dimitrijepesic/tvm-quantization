/*
 * Microbenchmark + self-contained PID 1 init for raspi4b QEMU run.
 * Statically linked aarch64 binary that:
 *   1. mounts /proc and /sys,
 *   2. dumps /proc/cpuinfo (proves Cortex-A72 ISA on raspi4b machine),
 *   3. runs fp32 / int32 / int16 / int8 MAC-throughput loops,
 *   4. prints throughput ratios relative to fp32,
 *   5. syncs and powers off the VM (so QEMU exits cleanly).
 *
 * Build:
 *   aarch64-linux-gnu-gcc -O2 -mcpu=cortex-a72 -static -o microbench microbench.c
 *
 * Run:
 *   qemu-system-aarch64 -M raspi4b ... -initrd initramfs.cpio.gz
 *   (cpio archive contains /init = this binary, plus empty /proc and /sys dirs)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <linux/reboot.h>

#define BUF 4096
#define ITERS 200000L

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static double bench_fp32(void) {
    static float x[BUF], y[BUF];
    for (int i = 0; i < BUF; i++) {
        x[i] = (float)(i % 7 + 1) * 0.5f;
        y[i] = (float)((i + 3) % 5 + 1) * 0.25f;
    }
    float acc = 0;
    double t0 = now_sec();
    for (long it = 0; it < ITERS; it++) {
        float a = 0;
        for (int i = 0; i < BUF; i++) a += x[i] * y[i];
        acc += a * 1e-6f;
    }
    double t1 = now_sec();
    double s = t1 - t0;
    double mops = (double)ITERS * BUF / s / 1e6;
    printf("%-7s  time=%7.3fs  MAC_MOPs=%8.2f  sink=%.4e\n", "fp32", s, mops, (double)acc);
    fflush(stdout);
    return mops;
}

static double bench_int32(void) {
    static int32_t x[BUF], y[BUF];
    for (int i = 0; i < BUF; i++) { x[i] = (i % 7) + 1; y[i] = ((i + 3) % 5) + 1; }
    int32_t acc = 0;
    double t0 = now_sec();
    for (long it = 0; it < ITERS; it++) {
        int32_t a = 0;
        for (int i = 0; i < BUF; i++) a += x[i] * y[i];
        acc ^= a;
    }
    double t1 = now_sec();
    double s = t1 - t0;
    double mops = (double)ITERS * BUF / s / 1e6;
    printf("%-7s  time=%7.3fs  MAC_MOPs=%8.2f  sink=%d\n", "int32", s, mops, acc);
    fflush(stdout);
    return mops;
}

static double bench_int16(void) {
    static int16_t x[BUF], y[BUF];
    for (int i = 0; i < BUF; i++) { x[i] = (int16_t)((i % 7) + 1); y[i] = (int16_t)(((i + 3) % 5) + 1); }
    int32_t acc = 0;
    double t0 = now_sec();
    for (long it = 0; it < ITERS; it++) {
        int32_t a = 0;
        for (int i = 0; i < BUF; i++) a += (int32_t)x[i] * (int32_t)y[i];
        acc ^= a;
    }
    double t1 = now_sec();
    double s = t1 - t0;
    double mops = (double)ITERS * BUF / s / 1e6;
    printf("%-7s  time=%7.3fs  MAC_MOPs=%8.2f  sink=%d\n", "int16", s, mops, acc);
    fflush(stdout);
    return mops;
}

static double bench_int8(void) {
    static int8_t x[BUF], y[BUF];
    for (int i = 0; i < BUF; i++) { x[i] = (int8_t)((i % 7) + 1); y[i] = (int8_t)(((i + 3) % 5) + 1); }
    int32_t acc = 0;
    double t0 = now_sec();
    for (long it = 0; it < ITERS; it++) {
        int32_t a = 0;
        for (int i = 0; i < BUF; i++) a += (int32_t)x[i] * (int32_t)y[i];
        acc ^= a;
    }
    double t1 = now_sec();
    double s = t1 - t0;
    double mops = (double)ITERS * BUF / s / 1e6;
    printf("%-7s  time=%7.3fs  MAC_MOPs=%8.2f  sink=%d\n", "int8", s, mops, acc);
    fflush(stdout);
    return mops;
}

static void dump_cpuinfo(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) { perror("open /proc/cpuinfo"); return; }
    char line[512];
    int printed = 0;
    while (fgets(line, sizeof line, f)) {
        if (strstr(line, "processor") || strstr(line, "model name") ||
            strstr(line, "CPU implementer") || strstr(line, "CPU part") ||
            strstr(line, "CPU variant") || strstr(line, "CPU revision") ||
            strstr(line, "Features") || strstr(line, "Hardware")) {
            fputs(line, stdout);
            printed++;
        }
        if (printed > 60) break;
    }
    fclose(f);
}

int main(void) {
    /* As PID 1 in initramfs we must set up /dev, /proc, /sys ourselves.
     * The kernel could not open an initial console (no /dev/console node
     * in the initramfs), so fd 0/1/2 are closed. Re-open them on
     * /dev/kmsg after mounting devtmpfs -- writes there route into the
     * kernel log and out the serial console. */
    mkdir("/dev",  0755);
    mkdir("/proc", 0555);
    mkdir("/sys",  0555);
    mount("devtmpfs", "/dev",  "devtmpfs", 0, NULL);
    int kfd = open("/dev/kmsg", O_WRONLY);
    if (kfd < 0) kfd = open("/dev/console", O_WRONLY);
    if (kfd >= 0) { dup2(kfd, 0); dup2(kfd, 1); dup2(kfd, 2); if (kfd > 2) close(kfd); }
    /* Reset stdio FILE* streams so libc reopens them on the new fds. */
    freopen("/dev/kmsg", "w", stdout);
    freopen("/dev/kmsg", "w", stderr);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    mount("proc",  "/proc", "proc",  0, NULL);
    mount("sysfs", "/sys",  "sysfs", 0, NULL);
    /* Emit a heartbeat first thing so we know the redirection worked. */
    const char hb[] = "<6>microbench: init started, kmsg redirection active\n";
    write(1, hb, sizeof(hb)-1);

    printf("\n=========================================================\n");
    printf("  raspi4b QEMU microbenchmark (PID 1 initramfs init)\n");
    printf("=========================================================\n\n");
    fflush(stdout);

    printf("== /proc/cpuinfo (filtered) ==\n");
    dump_cpuinfo();
    printf("\n");
    fflush(stdout);

    printf("== Microbenchmark on emulated Cortex-A72 ==\n");
    printf("BUF=%d  ITERS=%ld  total_ops_per_dtype=%lld\n\n",
           BUF, ITERS, (long long)ITERS * BUF);
    fflush(stdout);

    double m_fp32 = bench_fp32();
    double m_i32  = bench_int32();
    double m_i16  = bench_int16();
    double m_i8   = bench_int8();

    printf("\n== Throughput ratios (relative to fp32) ==\n");
    printf("fp32 :  1.00x\n");
    printf("int32:  %.2fx\n", m_i32 / m_fp32);
    printf("int16:  %.2fx\n", m_i16 / m_fp32);
    printf("int8 :  %.2fx\n", m_i8  / m_fp32);
    printf("\n=== BENCHMARK COMPLETE ===\n");
    fflush(stdout);

    sync();
    /* Halt the VM so QEMU exits cleanly and we can read the serial log. */
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    /* If poweroff returns (it shouldn't), loop forever instead of panicking. */
    for (;;) pause();
    return 0;
}
