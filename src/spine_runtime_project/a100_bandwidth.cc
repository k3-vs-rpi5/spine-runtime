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

/* The four-bit access pattern the vendor's i8i4 kernels use: 16 bytes of packed
 * weights plus one 2-byte scale per 32 weights, with all the packed data first
 * and all the scales in a second region. This is the variant that decides
 * whether the decode path's 86.8 % of the ceiling is an instruction-mix limit
 * or a layout limit: it moves the same bytes as a plain stream but has to
 * unpack every nibble and fetch a scale every 32 weights. */
void read_kernel_packed(spert::Context* context, const uint8_t* data, uint64_t bytes,
                        uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t per_tile = bytes / tiles;            /* packed data bytes */
    const uint64_t blocks = per_tile / 16;              /* 16 data bytes per 32 weights */
    const uint8_t* base = data + static_cast<uint64_t>(tile) * per_tile;
    const uint8_t* scales = data + bytes + static_cast<uint64_t>(tile) * blocks * 2;
    uint64_t accumulator = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t block = 0; block < blocks; ++block) {
            const uint8_t* packed = base + block * 16;
            uint64_t low = 0, high = 0;
            uint16_t scale = 0;

            __builtin_memcpy(&low, packed, sizeof(low));
            __builtin_memcpy(&high, packed + 8, sizeof(high));
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            /* Every byte of both words: a 64-bit word holds sixteen nibbles, and
             * an earlier version of this loop walked four shift values and
             * summed only a quarter of the weights while still reading every
             * byte. The host checksum caught it, which is exactly why a kernel
             * is checked against a reference before its rate is quoted. */
            for (unsigned shift = 0; shift < 64; shift += 8) {
                const uint8_t low_byte = (uint8_t)(low >> shift);
                const uint8_t high_byte = (uint8_t)(high >> shift);

                accumulator += (low_byte & 0x0fu) + (low_byte >> 4);
                accumulator += (high_byte & 0x0fu) + (high_byte >> 4);
            }
            accumulator += scale;
        }
    out[tile] = accumulator;
}

uint64_t host_checksum_packed(const uint8_t* data, uint64_t bytes, uint32_t passes) {
    const uint64_t blocks = bytes / 16;
    const uint8_t* scales = data + bytes;
    uint64_t total = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t block = 0; block < blocks; ++block) {
            uint16_t scale = 0;

            for (uint64_t index = 0; index < 16; ++index) {
                const uint8_t byte = data[block * 16 + index];

                total += (byte & 0x0fu) + (byte >> 4);
            }
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            total += scale;
        }
    return total;
}

/* The vendor's decode path reads its weights in calls that average 65,792 bytes
 * (282,805 MiB over 4,506,936 calls), scattered across a 1.5 GiB weight set
 * rather than one contiguous run. This variant reproduces exactly that: the
 * buffer is cut into 65,792-byte tiles, the tiles are visited in a permuted
 * order, and a working set of the model's size can be requested. If the rate
 * falls to the ~87 % the decode path shows, the missing 13 % is tile
 * granularity and working-set size, not the kernel's arithmetic. */
constexpr uint64_t kTileBytes = 65792;

void read_kernel_tiles(spert::Context* context, const uint8_t* data, uint64_t bytes,
                       uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t count = bytes / kTileBytes;
    uint64_t stride = 8191;
    uint64_t accumulator = 0;

    if (count == 0) {
        out[tile] = 0;
        return;
    }
    /* A permutation: the multiplier has to be coprime with the tile count, or a
     * tile would be read twice and another never, which the checksum rejects. */
    for (;;) {
        uint64_t a = stride, b = count;

        while (b != 0) {
            const uint64_t next = a % b;

            a = b;
            b = next;
        }
        if (a == 1)
            break;
        stride++;
    }
    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t index = tile; index < count; index += tiles) {
            const uint64_t where = (index * stride) % count;
            const uint64_t* base =
                reinterpret_cast<const uint64_t*>(data + where * kTileBytes);

            for (uint64_t word = 0; word < kTileBytes / sizeof(uint64_t); ++word)
                accumulator += base[word];
        }
    out[tile] = accumulator;
}

uint64_t host_checksum_tiles(const uint8_t* data, uint64_t bytes, uint32_t passes) {
    const uint64_t count = bytes / kTileBytes;
    uint64_t total = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t index = 0; index < count * (kTileBytes / 8); ++index)
            total += reinterpret_cast<const uint64_t*>(data)[index];
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

/* One weight read serving R activation rows.
 *
 * This is the question the vendor's decode path leaves open. Its one-row kernels
 * stream at 9.0 GiB/s and its four-row kernels at 5.8 GiB/s, so batching loses
 * weight bandwidth exactly when it is supposed to amortise the read. This kernel
 * keeps the read identical and lets the arithmetic grow with the row count: each
 * packed block is unpacked once and multiplied by R activations (here r+1, a
 * value that costs nothing to obtain and is exactly reproducible on the host, so
 * a fast-but-wrong kernel cannot pass).
 *
 * If the rate holds as R grows, the amortisation the vendor loses is recoverable
 * in principle and only its kernel structure prevents it. */
template <int R>
void read_kernel_gemv(spert::Context* context, const uint8_t* data, uint64_t bytes,
                      uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t per_tile = bytes / tiles;            /* packed data bytes */
    const uint64_t blocks = per_tile / 16;              /* 16 data bytes per 32 weights */
    const uint8_t* base = data + static_cast<uint64_t>(tile) * per_tile;
    const uint8_t* scales = data + bytes + static_cast<uint64_t>(tile) * blocks * 2;
    uint64_t accumulator = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t block = 0; block < blocks; ++block) {
            const uint8_t* packed = base + block * 16;
            uint64_t low = 0, high = 0;
            uint16_t scale = 0;
            uint64_t nibbles = 0;
            uint64_t scaled;

            __builtin_memcpy(&low, packed, sizeof(low));
            __builtin_memcpy(&high, packed + 8, sizeof(high));
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            for (unsigned shift = 0; shift < 64; shift += 8) {
                const uint8_t low_byte = (uint8_t)(low >> shift);
                const uint8_t high_byte = (uint8_t)(high >> shift);

                nibbles += (low_byte & 0x0fu) + (low_byte >> 4);
                nibbles += (high_byte & 0x0fu) + (high_byte >> 4);
            }
            scaled = nibbles * scale;
            /* R rows consume the same read; the row weights are what makes the
             * arithmetic grow without touching memory again. */
            for (int row = 0; row < R; ++row)
                accumulator += scaled * static_cast<uint64_t>(row + 1);
        }
    out[tile] = accumulator;
}

template <int R>
uint64_t host_checksum_gemv(const uint8_t* data, uint64_t bytes, uint32_t passes) {
    const uint64_t blocks = bytes / 16;
    const uint8_t* scales = data + bytes;
    uint64_t total = 0;
    uint64_t row_weight = 0;

    for (int row = 0; row < R; ++row)
        row_weight += static_cast<uint64_t>(row + 1);
    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t block = 0; block < blocks; ++block) {
            uint16_t scale = 0;
            uint64_t nibbles = 0;

            for (uint64_t index = 0; index < 16; ++index) {
                const uint8_t byte = data[block * 16 + index];

                nibbles += (byte & 0x0fu) + (byte >> 4);
            }
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            total += nibbles * scale * row_weight;
        }
    return total;
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
    enum class kind { bytes, words, packed, tiled, gemv1, gemv4, gemv8 };
    const struct {
        const char* name;
        kernel_fn kernel;
        kind how;
    } variants[] = {
        {"byte", read_kernel, kind::bytes},
        {"word", read_kernel_words, kind::words},
        {"pack4", read_kernel_packed, kind::packed},
        {"tiled", read_kernel_tiles, kind::tiled},
        /* the same packed read, serving 1, 4 and 8 rows from one pass over it */
        {"gemvR1", read_kernel_gemv<1>, kind::gemv1},
        {"gemvR4", read_kernel_gemv<4>, kind::gemv4},
        {"gemvR8", read_kernel_gemv<8>, kind::gemv8},
    };

    std::printf("%10s %6s %8s %8s %10s %12s %s\n", "size_MiB", "kernel", "cores",
                "passes", "seconds", "MiB_per_s", "checksum");
    for (size_t size_mib : sizes) {
        size_t bytes = size_mib * kMiB;
        size_t scale_bytes = bytes / 8;          /* 2 bytes per 16 data bytes */
        bool huge = false;
        uint8_t* buffer = allocate(bytes + scale_bytes, &huge);

        if (buffer == nullptr) {
            std::printf("%10zu %8s %8s %10s %12s %s\n", size_mib, "-", "-", "-", "-",
                        "mmap-failed");
            continue;
        }
        /* A pattern the compiler cannot fold into a constant. */
        for (size_t index = 0; index < bytes; ++index)
            buffer[index] = static_cast<uint8_t>((index * 31u + 7u) & 0xffu);
        for (size_t index = 0; index < scale_bytes; ++index)
            buffer[bytes + index] = static_cast<uint8_t>((index * 13u + 5u) & 0xffu);

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

            if (variant.how == kind::words) {
                const uint64_t* words = reinterpret_cast<const uint64_t*>(buffer);
                uint64_t count = bytes / sizeof(uint64_t);

                for (uint32_t pass = 0; pass < passes; ++pass)
                    for (uint64_t index = 0; index < count; ++index)
                        expected += words[index];
            } else if (variant.how == kind::packed) {
                expected = host_checksum_packed(buffer, bytes, passes);
            } else if (variant.how == kind::tiled) {
                expected = host_checksum_tiles(buffer, bytes, passes);
            } else if (variant.how == kind::gemv1) {
                expected = host_checksum_gemv<1>(buffer, bytes, passes);
            } else if (variant.how == kind::gemv4) {
                expected = host_checksum_gemv<4>(buffer, bytes, passes);
            } else if (variant.how == kind::gemv8) {
                expected = host_checksum_gemv<8>(buffer, bytes, passes);
            } else {
                expected = host_checksum(buffer, bytes, passes);
            }
            uint64_t got = 0;

            for (uint64_t value : out)
                got += value;
            /* The packed variant moves its scale region as well, so its traffic
             * is what the kernels would read: data plus scales. */
            double moved = static_cast<double>(bytes) +
                           (variant.how == kind::packed || variant.how == kind::gemv1 ||
                            variant.how == kind::gemv4 || variant.how == kind::gemv8
                                ? static_cast<double>(scale_bytes)
                                : 0.0);
            if (variant.how == kind::tiled)
                moved = static_cast<double>((bytes / kTileBytes) * kTileBytes);
            double mib_per_s = seconds > 0.0 ? moved * passes / kMiB / seconds : 0.0;
            std::printf("%10zu %6s %8u %8u %10.4f %12.1f %s\n", size_mib,
                        variant.name, cores, passes, seconds, mib_per_s,
                        got == expected ? "ok" : "MISMATCH");
            std::printf("RESULT size_mib=%zu kernel=%s cores=%u passes=%u seconds=%.4f mib_per_s=%.1f moved_mib=%.1f hugepages=%d checksum=%s\n",
                        size_mib, variant.name, cores, passes, seconds, mib_per_s,
                        moved / kMiB, huge ? 1 : 0,
                        got == expected ? "ok" : "mismatch");
        }
        munmap(buffer, bytes + scale_bytes);
    }
    return 0;
}
