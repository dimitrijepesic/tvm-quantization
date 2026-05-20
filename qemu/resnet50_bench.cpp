/*
 * resnet50_bench.cpp -- PID 1 init for QEMU raspi4b that loads the
 * cross-compiled TVM ResNet50 .so + .json + .params triples, runs a
 * warmup + timed inference for both the fp32 and int8 builds, prints
 * results via /dev/kmsg (serial console), then powers off so QEMU
 * exits cleanly.
 *
 * Build (host):
 *   aarch64-linux-gnu-g++ -std=c++17 -O2 -mcpu=cortex-a72 \
 *       -I$TVM/include -I$TVM/3rdparty/dlpack/include -I$TVM/3rdparty/dmlc-core/include \
 *       -DDMLC_USE_LOGGING_LIBRARY='<tvm/runtime/logging.h>' \
 *       -static-libstdc++ -static-libgcc \
 *       resnet50_bench.cpp $TVM/build_arm64/libtvm_runtime.a \
 *       -lpthread -ldl -lm -lrt \
 *       -o resnet50_bench
 *
 * Layout expected inside the running guest (all in /):
 *   /resnet50_fp32_arm.so   /resnet50_fp32_arm.json   /resnet50_fp32_arm.params
 *   /resnet50_int8_arm.so   /resnet50_int8_arm.json   /resnet50_int8_arm.params
 */

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/ndarray.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <linux/reboot.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static double now_sec() {
    using clk = std::chrono::steady_clock;
    auto t = clk::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

static std::string slurp(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        return std::string();
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void setup_init_environment() {
    /* Set up minimal filesystems, then redirect stdio to /dev/kmsg so
     * printf/fprintf go to the kernel log -> serial console. */
    mkdir("/dev",  0755);
    mkdir("/proc", 0555);
    mkdir("/sys",  0555);
    mount("devtmpfs", "/dev",  "devtmpfs", 0, nullptr);
    int kfd = open("/dev/kmsg", O_WRONLY);
    if (kfd < 0) kfd = open("/dev/console", O_WRONLY);
    if (kfd >= 0) {
        dup2(kfd, 0); dup2(kfd, 1); dup2(kfd, 2);
        if (kfd > 2) close(kfd);
    }
    std::freopen("/dev/kmsg", "w", stdout);
    std::freopen("/dev/kmsg", "w", stderr);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    mount("proc",  "/proc", "proc",  0, nullptr);
    mount("sysfs", "/sys",  "sysfs", 0, nullptr);
    const char hb[] = "<6>resnet50_bench: init started, kmsg redirection active\n";
    (void)!write(1, hb, sizeof(hb) - 1);
}

static void poweroff_and_hang() {
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    for (;;) pause();
}

struct BenchResult {
    double warmup_s;
    double runs[5];
    double mean_s;
};

static BenchResult bench_one(const std::string& tag,
                             const std::string& so_path,
                             const std::string& json_path,
                             const std::string& params_path) {
    BenchResult r{};
    std::printf("\n---- %s ----\n", tag.c_str());
    std::printf("loading %s\n", so_path.c_str());
    tvm::runtime::Module mod_factory =
        tvm::runtime::Module::LoadFromFile(so_path);

    DLDevice dev{kDLCPU, 0};
    std::string json_str = slurp(json_path.c_str());
    if (json_str.empty()) {
        std::printf("%s: missing json, skipping\n", tag.c_str());
        return r;
    }

    /* The standard recipe is to call the factory's `default` function with
     * the device; alternatively the graph_executor.create registry entry
     * takes (json_str, mod_factory, device_type, device_id). */
    tvm::runtime::PackedFunc create =
        *tvm::runtime::Registry::Get("tvm.graph_executor.create");
    tvm::runtime::Module gmod = create(json_str, mod_factory,
                                       (int)kDLCPU, (int)0);

    /* Load params bytes. */
    std::string params_str = slurp(params_path.c_str());
    if (params_str.empty()) {
        std::printf("%s: missing params, skipping\n", tag.c_str());
        return r;
    }
    TVMByteArray params_ba;
    params_ba.data = params_str.data();
    params_ba.size = params_str.size();
    tvm::runtime::PackedFunc load_params = gmod.GetFunction("load_params");
    load_params(params_ba);

    /* Build input tensor x of shape [1,3,224,224] filled with a small
     * deterministic pattern (avoid Inf/NaN poison). */
    DLDataType f32{kDLFloat, 32, 1};
    tvm::runtime::NDArray x =
        tvm::runtime::NDArray::Empty({1, 3, 224, 224}, f32, dev);
    float* xp = static_cast<float*>(x->data);
    for (int i = 0; i < 1 * 3 * 224 * 224; ++i) {
        xp[i] = (float)((i % 251) - 125) * (1.0f / 128.0f);
    }
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    set_input("input0", x);

    tvm::runtime::PackedFunc run = gmod.GetFunction("run");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");

    /* Warmup. */
    double t0 = now_sec();
    run();
    double t1 = now_sec();
    r.warmup_s = t1 - t0;
    std::printf("%s warmup: %.3fs\n", tag.c_str(), r.warmup_s);

    /* Timed runs. */
    double sum = 0;
    for (int i = 0; i < 5; ++i) {
        double a = now_sec();
        run();
        double b = now_sec();
        r.runs[i] = b - a;
        sum += r.runs[i];
        std::printf("%s run[%d]: %.3fs\n", tag.c_str(), i, r.runs[i]);
    }
    r.mean_s = sum / 5.0;
    std::printf("%s MEAN over 5 runs: %.3fs (warmup excluded)\n",
                tag.c_str(), r.mean_s);

    /* Sanity: read argmax to confirm execution actually produced output. */
    tvm::runtime::NDArray y =
        tvm::runtime::NDArray::Empty({1, 1000}, f32, dev);
    get_output(0, y);
    float* yp = static_cast<float*>(y->data);
    int amax = 0;
    float vmax = yp[0];
    for (int i = 1; i < 1000; ++i) if (yp[i] > vmax) { vmax = yp[i]; amax = i; }
    std::printf("%s output argmax=%d  value=%.4f\n", tag.c_str(), amax, vmax);

    return r;
}

int main() {
    setup_init_environment();

    std::printf("\n=========================================================\n");
    std::printf("  raspi4b ResNet50 inference (TVM C++ runtime, PID 1)\n");
    std::printf("=========================================================\n");

    BenchResult rfp32, rint8;
    try {
        rfp32 = bench_one("fp32",
                          "/resnet50_fp32_arm.so",
                          "/resnet50_fp32_arm.json",
                          "/resnet50_fp32_arm.params");
    } catch (const std::exception& e) {
        std::printf("fp32 FAILED: %s\n", e.what());
    } catch (...) {
        std::printf("fp32 FAILED: unknown exception\n");
    }

    try {
        rint8 = bench_one("int8",
                          "/resnet50_int8_arm.so",
                          "/resnet50_int8_arm.json",
                          "/resnet50_int8_arm.params");
    } catch (const std::exception& e) {
        std::printf("int8 FAILED: %s\n", e.what());
    } catch (...) {
        std::printf("int8 FAILED: unknown exception\n");
    }

    std::printf("\n=========================================================\n");
    std::printf("  SUMMARY (raspi4b, emulated Cortex-A72, TCG)\n");
    std::printf("=========================================================\n");
    std::printf("fp32 mean : %.3f s (warmup %.3f s)\n",
                rfp32.mean_s, rfp32.warmup_s);
    std::printf("int8 mean : %.3f s (warmup %.3f s)\n",
                rint8.mean_s, rint8.warmup_s);
    if (rfp32.mean_s > 0 && rint8.mean_s > 0) {
        std::printf("int8 / fp32 ratio: %.3fx", rint8.mean_s / rfp32.mean_s);
        if (rint8.mean_s < rfp32.mean_s) std::printf("  (int8 FASTER)");
        else std::printf("  (int8 slower)");
        std::printf("\n");
    }
    std::printf("=== BENCHMARK COMPLETE ===\n");

    poweroff_and_hang();
    return 0;
}
