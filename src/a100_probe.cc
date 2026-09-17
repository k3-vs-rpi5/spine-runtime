// SPDX-License-Identifier: Apache-2.0
/*
 * A100 bring-up probe.
 *
 * Purpose: prove, with our own code, that this workspace can compile work for
 * the AI cluster through the documented runtime and that the work really runs
 * there. Two questions are answered in one run:
 *
 *   1. which backends this board exposes, and what each one grants
 *      (`backend_info()`: core count, per-core shared buffer, vector length,
 *      architecture id);
 *   2. whether a tile kernel submitted through `spert::Stream` produces the
 *      same answer as the host-side reference, and how long it takes.
 *
 * The TCM section uses the board's own `spine_tcm` library in its header-only
 * loader mode, so the binary depends on `libspert` being next to it and nothing
 * else: if the driver is missing, availability is reported as zero rather than
 * turning into a link-time failure.
 */
#include "spert.hpp"
#include "spine_tcm.h"

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kTiles = 8;
constexpr uint32_t kElements = 1u << 16;
constexpr int kDefaultIterations = 50;

/* Every tile sums the elements it owns; the host computes the same sums. */
void sum_kernel(spert::Context* context, const uint8_t* data, uint32_t* out,
                uint32_t elements) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    uint32_t accumulator = 0;

    for (uint32_t index = tile; index < elements; index += tiles)
        accumulator += data[index];
    out[tile] = accumulator;
}

void describe_backend(const char* name, spert::BackendHandle handle) {
    if (!handle.valid()) {
        std::printf("backend %-18s unavailable\n", name);
        return;
    }
    spert::BackendInfo info = spert::backend_info(handle);

    std::printf("backend %-18s cores=%zu shared_buffer=%zu bytes vlen=%zu arch_id=0x%llx\n",
                name, info.num_cores, info.shared_mem_size, info.vlen,
                static_cast<long long>(info.core_arch_id));
}

void report_tcm() {
    std::printf("tcm: version=%s available=%d\n",
                spine_tcm_version() ? spine_tcm_version() : "(none)",
                spine_tcm_is_available());
    if (!spine_tcm_is_available()) {
        std::printf("RESULT tcm=unavailable\n");
        return;
    }
    for (int id = 0; id < 8; ++id) {
        spine_tcm_block_info_t info;

        std::memset(&info, 0, sizeof(info));
        if (spine_tcm_block_info(id, &info) == 0)
            std::printf("tcm block %d: size=%zu affinity=0x%llx acquired=%d owner=%d\n", id,
                        info.size, static_cast<unsigned long long>(info.cpu_affinity_mask),
                        info.is_acquired, info.owner_tid);
    }
    void* buffer = spine_tcm_mem_get(0);

    std::printf("tcm: mem_get(0)=%s\n", buffer ? "granted" : "refused");
    if (buffer) {
        volatile uint8_t* byte = static_cast<volatile uint8_t*>(buffer);

        byte[0] = 0x5a;
        std::printf("tcm: wrote and read back 0x%02x at the first byte\n", byte[0]);
        std::printf("tcm: mem_free(0)=%d\n", spine_tcm_mem_free(0));
    }
    std::printf("RESULT tcm=%s\n", buffer ? "granted" : "refused");
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = kDefaultIterations;

    for (int index = 1; index < argc; ++index)
        if (std::strcmp(argv[index], "--iterations") == 0 && index + 1 < argc)
            iterations = std::atoi(argv[++index]);
    if (iterations < 1)
        iterations = 1;
    std::printf("spine-runtime probe: start\n");
    std::printf("probe: iterations=%d\n", iterations);
    describe_backend("default", spert::default_backend());
    describe_backend("spacemit-k3", spert::backend("spacemit-k3"));
    describe_backend("spacemit-k3-x100", spert::backend("spacemit-k3-x100"));
    describe_backend("generic/qemu", spert::backend("generic/qemu"));

    spert::BackendHandle k3 = spert::backend("spacemit-k3");

    if (!k3.valid()) {
        std::printf("RESULT kernels=skipped-reason=a100-backend-unavailable\n");
        report_tcm();
        return 3;
    }

    spert::StreamConfig config;

    config.n_cores = kTiles;
    config.core_ids = {8, 9, 10, 11, 12, 13, 14, 15};
    spert::Stream stream(k3, config);
    if (!stream.valid()) {
        std::printf("RESULT kernels=skipped-reason=stream-refused\n");
        report_tcm();
        return 4;
    }
    std::printf("stream: requested %u cores (ids 8-15), granted %u\n", kTiles,
                stream.core_count());

    std::vector<uint8_t> data(kElements);
    std::vector<uint32_t> result(kTiles, 0);

    for (uint32_t index = 0; index < kElements; ++index)
        data[index] = static_cast<uint8_t>((index * 7u + 3u) & 0xffu);

    auto launch = [&]() {
        spert::Future future =
            stream.launch(spert::Grid(kTiles), sum_kernel, data.data(), result.data(),
                          kElements);
        return future.valid() ? future.sync() : spert::Status::InvalidArg;
    };
    spert::Status status = launch();

    if (status != spert::Status::Ok) {
        std::printf("RESULT kernels=skipped-reason=launch-failed status=%s\n",
                    spert::to_string(status));
        report_tcm();
        return 5;
    }

    std::vector<uint32_t> expected(kTiles, 0);

    for (uint32_t index = 0; index < kElements; ++index)
        expected[index % kTiles] += data[index];

    int mismatches = 0;

    for (uint32_t tile = 0; tile < kTiles; ++tile)
        if (result[tile] != expected[tile]) {
            if (mismatches < 3)
                std::printf("kernel mismatch at tile %u: %u != %u\n", tile, result[tile],
                            expected[tile]);
            mismatches++;
        }
    auto begin = std::chrono::steady_clock::now();

    for (int iteration = 0; iteration < iterations; ++iteration)
        if (launch() != spert::Status::Ok) {
            std::printf("RESULT kernels=skipped-reason=iteration-failed\n");
            report_tcm();
            return 6;
        }
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    std::printf("kernel: %u tiles x %u elements, %d iterations in %.3f s = %.1f us/launch\n",
                kTiles, kElements, iterations, elapsed, elapsed * 1e6 / iterations);
    std::printf("RESULT kernels=%s mismatches=%d us_per_launch=%.1f\n",
                mismatches == 0 ? "ok" : "mismatch", mismatches,
                elapsed * 1e6 / iterations);
    report_tcm();
    return mismatches == 0 ? 0 : 7;
}
