// SPDX-License-Identifier: Apache-2.0
/*
 * AI-cluster memory-bandwidth microbenchmark.
 *
 * Why: every decode conclusion so far leans on a memory roof of 21,160 MiB/s
 * that was measured on the X100 cores. Nobody has ever measured what the AI
 * cluster's own memory path can do, so "decode is at the roof" is an
 * assumption, not a result. This program measures it with a streaming read
 * kernel of known size, submitted through the same documented runtime that the
 * vendor's LLM backend uses.
 *
 * Method: for each buffer size, the grid gives every compute core one
 * contiguous chunk and the kernel reads every byte of it `passes` times,
 * accumulating a sum. Bytes moved are therefore exactly
 * `size x passes`, the wall clock is taken around `Future::sync()`, and the
 * checksum is compared against a host-side reference so a fast-but-wrong kernel
 * cannot pass as a fast one.
 *
 *   a100_bandwidth [size_mib ...]        default: 8 64 256
 */
#include "spert.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {

constexpr size_t kMiB = 1024 * 1024;
constexpr uint32_t kMinPasses = 8;
constexpr double kMinSeconds = 0.05;

/* Reads every byte the tile owns, `passes` times. The sum is what keeps the
 * loads from being optimised away, and summing bytes (not 64-byte strides)
 * means the bytes moved are exactly the bytes asked for. */
void read_kernel(spert::Context* context, const uint8_t* data, uint64_t bytes,
                 uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t per_tile = bytes / tiles;
    const uint8_t* base = data + static_cast<uint64_t>(tile) * per_tile;
    uint64_t accumulator = 0;

    for (uint32_t pass = 0; pass < passes; ++pass) {
        uint64_t index = 0;

        for (; index + 8 <= per_tile; index += 8)
            accumulator += static_cast<uint64_t>(base[index]) + base[index + 1] +
                           base[index + 2] + base[index + 3] + base[index + 4] +
                           base[index + 5] + base[index + 6] + base[index + 7];
        for (; index < per_tile; ++index)
            accumulator += base[index];
    }
    out[tile] = accumulator;
}

uint64_t host_checksum(const uint8_t* data, uint64_t bytes, uint32_t passes) {
    uint64_t total = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t index = 0; index < bytes; ++index)
            total += data[index];
    return total;
}

/* The same read, one 64-bit word at a time. The byte kernel above issues eight
 * loads per eight bytes, so it measures how fast this core can issue byte
 * loads, not how fast memory can feed it; this variant issues one load per
 * eight bytes, which is what a streaming weight read looks like. */
void read_kernel_words(spert::Context* context, const uint8_t* data, uint64_t bytes,
                       uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t per_tile = bytes / tiles / sizeof(uint64_t);
    const uint64_t* base =
        reinterpret_cast<const uint64_t*>(data) +
        static_cast<uint64_t>(tile) * per_tile;
    uint64_t accumulator = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t index = 0; index < per_tile; ++index)
            accumulator += base[index];
    out[tile] = accumulator;
}

uint8_t* allocate(size_t bytes, bool* huge) {
    const size_t page = 2 * kMiB;
    size_t rounded = (bytes + page - 1) / page * page;
    void* buffer = mmap(nullptr, rounded, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (buffer == MAP_FAILED)
        return nullptr;
    *huge = madvise(buffer, rounded, MADV_HUGEPAGE) == 0;
    return static_cast<uint8_t*>(buffer);
}

void usage() {
    std::fprintf(stderr, "usage: a100_bandwidth [size_mib ...]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<size_t> sizes;

    if (argc == 2 && (std::strcmp(argv[1], "-h") == 0 ||
                      std::strcmp(argv[1], "--help") == 0)) {
        usage();
        return 0;
    }
    for (int index = 1; index < argc; ++index) {
        long value = std::strtol(argv[index], nullptr, 10);

        if (value <= 0) {
            usage();
            return 2;
        }
        sizes.push_back(static_cast<size_t>(value));
    }
    if (sizes.empty()) {
        sizes = {8, 64, 256};
    }

    spert::BackendHandle k3 = spert::backend("spacemit-k3");

    if (!k3.valid())
        k3 = spert::default_backend();
    spert::BackendInfo info = spert::backend_info(k3);

    std::printf("backend default cores=%zu shared_buffer=%zu bytes vlen=%zu arch_id=0x%llx\n",
                info.num_cores, info.shared_mem_size, info.vlen,
                static_cast<long long>(info.core_arch_id));
    if (info.num_cores == 0) {
        std::printf("RESULT bandwidth=unavailable reason=no-backend\n");
        return 1;
    }

    spert::StreamConfig config;

    config.n_cores = static_cast<uint32_t>(info.num_cores);
    config.core_ids = {8, 9, 10, 11, 12, 13, 14, 15};
    spert::Stream stream(k3, config);
    if (!stream.valid()) {
        std::printf("RESULT bandwidth=unavailable reason=stream-refused\n");
        return 1;
    }
    const uint32_t cores = stream.core_count();
    std::vector<uint64_t> out(cores, 0);

    std::printf("stream: requested %zu cores (ids 8-15), granted %u\n", info.num_cores,
                cores);

    using kernel_fn = void (*)(spert::Context*, const uint8_t*, uint64_t, uint32_t,
                               uint64_t*);
    const struct {
        const char* name;
        kernel_fn kernel;
        bool words;
    } variants[] = {
        {"byte", read_kernel, false},
        {"word", read_kernel_words, true},
    };

    std::printf("%10s %6s %8s %8s %10s %12s %s\n", "size_MiB", "kernel", "cores",
                "passes", "seconds", "MiB_per_s", "checksum");
    for (size_t size_mib : sizes) {
        size_t bytes = size_mib * kMiB;
        bool huge = false;
        uint8_t* buffer = allocate(bytes, &huge);

        if (buffer == nullptr) {
            std::printf("%10zu %8s %8s %10s %12s %s\n", size_mib, "-", "-", "-", "-",
                        "mmap-failed");
            continue;
        }
        /* A pattern the compiler cannot fold into a constant. */
        for (size_t index = 0; index < bytes; ++index)
            buffer[index] = static_cast<uint8_t>((index * 31u + 7u) & 0xffu);

        for (const auto& variant : variants) {
            uint32_t passes = kMinPasses;
            double seconds = 0.0;

            for (int attempt = 0; attempt < 4; ++attempt) {
                std::fill(out.begin(), out.end(), 0);
                auto start = std::chrono::steady_clock::now();
                auto future = stream.launch(spert::Grid{cores}, variant.kernel,
                                            static_cast<const uint8_t*>(buffer),
                                            static_cast<uint64_t>(bytes), passes,
                                            out.data());
                future.sync();
                auto end = std::chrono::steady_clock::now();

                seconds = std::chrono::duration<double>(end - start).count();
                if (seconds >= kMinSeconds)
                    break;
                passes *= 4;
            }

            uint64_t expected = 0;

            if (variant.words) {
                const uint64_t* words = reinterpret_cast<const uint64_t*>(buffer);
                uint64_t count = bytes / sizeof(uint64_t);

                for (uint32_t pass = 0; pass < passes; ++pass)
                    for (uint64_t index = 0; index < count; ++index)
                        expected += words[index];
            } else {
                expected = host_checksum(buffer, bytes, passes);
            }
            uint64_t got = 0;

            for (uint64_t value : out)
                got += value;
            double mib_per_s = seconds > 0.0
                                   ? static_cast<double>(bytes) * passes / kMiB / seconds
                                   : 0.0;
            std::printf("%10zu %6s %8u %8u %10.4f %12.1f %s\n", size_mib,
                        variant.name, cores, passes, seconds, mib_per_s,
                        got == expected ? "ok" : "MISMATCH");
            std::printf("RESULT size_mib=%zu kernel=%s cores=%u passes=%u seconds=%.4f mib_per_s=%.1f hugepages=%d checksum=%s\n",
                        size_mib, variant.name, cores, passes, seconds, mib_per_s,
                        huge ? 1 : 0, got == expected ? "ok" : "mismatch");
        }
        munmap(buffer, bytes);
    }
    return 0;
}
