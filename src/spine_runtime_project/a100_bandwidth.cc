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

#if defined(__riscv_v) && __riscv_v >= 1000000
#include <riscv_vector.h>
#define HAVE_RVV 1
#endif

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

/* The same read with the scale applied to each block, which is what a
 * quantisation-aware kernel does: sum of nibbles times the block's scale. */
uint64_t host_checksum_packed_scaled(const uint8_t* data, uint64_t bytes, uint32_t passes) {
    const uint64_t blocks = bytes / 16;
    const uint8_t* scales = data + bytes;
    uint64_t total = 0;

    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t block = 0; block < blocks; ++block) {
            uint16_t scale = 0;
            uint64_t nibbles = 0;

            for (uint64_t index = 0; index < 16; ++index) {
                const uint8_t byte = data[block * 16 + index];

                nibbles += (byte & 0x0fu) + (byte >> 4);
            }
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            total += nibbles * (uint64_t)scale;
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

#ifdef HAVE_RVV
static uint8_t* allocate(size_t bytes, bool* huge);

/* Does this cluster implement the public `sf.vqmacc` family at all?
 *
 * The instruction is public in the toolchain (`xsfvqmaccdod` / `xsfvqmaccqoq`,
 * opcode 0x5b) but has no public semantic documentation, so the first question is
 * not "what does its layout look like" but "does the silicon execute it": an
 * unimplemented instruction traps, and a trap here is visible as a failed launch.
 * One instruction, one accumulator in the launch buffer, no layout assumptions. */
void probe_sfq_kernel(spert::Context* context, const uint8_t* data, uint64_t bytes,
                      uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    int32_t* acc = reinterpret_cast<int32_t*>(out + (uint64_t)tile * 16);

    (void)bytes;
    (void)passes;
    for (int index = 0; index < 8; ++index)
        acc[index] = 0;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2   \n\t"
        "vle32.v v28, (%[ACC])       \n\t"
        "vsetvli t0, zero, e8, m1    \n\t"
        "vle8.v  v0, (%[A])          \n\t"
        "vle8.v  v1, (%[B])          \n\t"
        "sf.vqmacc.4x8x4 v28, v0, v1 \n\t"
        "vsetvli t0, zero, e32, m2   \n\t"
        "vse32.v v28, (%[ACC])       \n\t"
        : : [A] "r"(data), [B] "r"(data + 16), [ACC] "r"(acc)
        : "cc", "t0", "v0", "v1", "v28", "memory");
}

/* One `smt.vmadotsu.hp` per core: the block-quantized family the vendor's own
 * decode kernel runs, in the operand form the specification gives
 * (`vd, vs1, vs2, v0/v1, imm3, i4`) and the assembler enforces (`v0`/`v1` only).
 *
 * The tile spans more registers than the prose states, so the probe makes the
 * answer layout-independent instead of guessing the extent: every vector register
 * carries the same nibble pattern (0x1111, so every Int4 product is 1x1), the
 * scale register carries FP16 1.0 in every lane, and the accumulators start at
 * zero. Summed over K = 32 that is 32.0 = 0x5000 in every FP16 output lane,
 * whichever lanes the instruction actually writes. A trap shows up the way
 * `sf.vqmacc` did: a launch that never returns.
 */
void probe_smthp_kernel(spert::Context* context, const uint8_t* data, uint64_t bytes,
                        uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    /* v24 and v25 are the FP16 destination pair the specification's Copies rule
     * allows at VLEN=128; both are stored, so a two-register tile is observed
     * rather than assumed away. */
    /* Windows start after the 1 KiB flat area the control kernel also writes,
     * so the two addressing conventions cannot overwrite each other. */
    uint16_t* acc = reinterpret_cast<uint16_t*>(out + 128 + (uint64_t)tile * 32);

    (void)bytes;
    (void)passes;
    __asm__ volatile(
        "vsetvli t0, zero, e16, m1, tu, mu   \n\t"
        "vle16.v v16, (%[ONES])              \n\t"
        "vmv.v.v v1, v16                     \n\t"
        "vmv.v.v v2, v16                     \n\t"
        "vmv.v.v v3, v16                     \n\t"
        "vmv.v.v v4, v16                     \n\t"
        "vmv.v.v v5, v16                     \n\t"
        "vmv.v.v v6, v16                     \n\t"
        "vmv.v.v v7, v16                     \n\t"
        "vmv.v.v v8, v16                     \n\t"
        "vmv.v.v v9, v16                     \n\t"
        "vmv.v.v v10, v16                    \n\t"
        "vmv.v.v v11, v16                    \n\t"
        "vmv.v.v v12, v16                    \n\t"
        "vmv.v.v v13, v16                    \n\t"
        "vmv.v.v v14, v16                    \n\t"
        "vmv.v.v v15, v16                    \n\t"
        "vmv.v.v v17, v16                    \n\t"
        "vmv.v.v v18, v16                    \n\t"
        "vmv.v.v v19, v16                    \n\t"
        "vmv.v.v v20, v16                    \n\t"
        "vmv.v.v v21, v16                    \n\t"
        "vmv.v.v v22, v16                    \n\t"
        "vmv.v.v v23, v16                    \n\t"
        "vmv.v.v v26, v16                    \n\t"
        "vmv.v.v v27, v16                    \n\t"
        "vmv.v.v v28, v16                    \n\t"
        "vmv.v.v v29, v16                    \n\t"
        "vmv.v.v v30, v16                    \n\t"
        "vmv.v.v v31, v16                    \n\t"
        "vle16.v v0, (%[SCALE])              \n\t"
        "vmv.v.i v24, 0                      \n\t"
        "vmv.v.i v25, 0                      \n\t"
        "vmv.v.x v31, %[TILE]                \n\t"
        "smt.vmadotsu.hp v24,v16,v16,v0,0,i4 \n\t"
        "vse16.v v24, (%[ACC])               \n\t"
        /* The store above is deliberately issued with no fresh vsetvli, and the
         * ones below are issued after one: if the matrix instruction leaves the
         * vector unit in a state a plain store cannot use, that shows up as the
         * first region empty and the rest full. */
        "vsetvli t0, zero, e16, m1, tu, mu   \n\t"
        "vse16.v v25, (%[ACC2])              \n\t"
        /* the tile id, written by plain RVV: a core whose stores do not land is
         * then visibly different from a core whose instruction result is zero */
        "vse16.v v31, (%[ID])                \n\t"
        : : [ONES] "r"(data + 16), [SCALE] "r"(data), [ACC] "r"(acc),
            [ACC2] "r"(acc + 8), [ID] "r"(acc + 16), [TILE] "r"(tile)
        : "cc", "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
          "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
          "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
          "v29", "v30", "v31", "memory");
    /* And the same tile id again as a plain 64-bit scalar store, the shape every
     * other kernel in this file uses: that separates "the core did not run" from
     * "the vector store did not land". */
    out[(uint64_t)tile * 32 + 24] = 0x1000ull + tile;
}

/* The control for `--smthp`: the same launch, the same buffers and the same
 * reading code with no matrix instruction in the kernel at all. Whatever this
 * reports is the harness; the difference between the two runs is the
 * instruction. */
void probe_smthp_control_kernel(spert::Context* context, const uint8_t* data,
                                uint64_t bytes, uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    uint16_t* acc = reinterpret_cast<uint16_t*>(out + 128 + (uint64_t)tile * 32);

    (void)data;
    (void)bytes;
    (void)passes;
    __asm__ volatile(
        "vsetvli t0, zero, e16, m1, tu, mu \n\t"
        "vmv.v.x v24, %[TILE]              \n\t"
        "vse16.v v24, (%[ACC])             \n\t"
        : : [ACC] "r"(acc), [TILE] "r"(tile)
        : "cc", "t0", "v24", "memory");
    out[(uint64_t)tile * 32 + 24] = 0x2000ull + tile;
    /* The addressing convention every other kernel in this file uses and the
     * `a100_probe` validates per tile: one uint64 per tile, no window. */
    out[tile] = 0x3000ull + tile;
}

/* An RVV gemv in the shape the vendor's own kernels use.
 *
 * Disassembling `spacemit_kernels::ime2::gemm_kernel_i8i4_m1` in the board's
 * libggml-cpu.so.0.16.0 shows standard RVV - `vl4r.v` segmented loads - around a
 * matrix core this file originally read as absent. That reading was an artefact:
 * the disassembler gates each opcode behind the file's `Tag_RISCV_arch` string,
 * the vendor library advertises no vendor extension, and `objdump -M max` decodes
 * what plain `objdump -d` printed as `.insn`. The kernel's inner block is
 * `smt.vnpack4.vv` (nibble packing of the 4-bit weights), `smt.vmadotsu.hp` and
 * `smt.vmadotu.hp` (block-quantized Int4/Int8 matrix MAC against FP16 scales into
 * FP16) and `smt.vpack.vv` (accumulator interleave), all public and all
 * assemblable here as of this writing. This RVV gemv remains the reference for
 * what ordinary vector code reaches on this cluster.
 *
 * This kernel streams the same 4-bit block format the other variants use (16 data
 * bytes + a 2-byte scale per 32 weights), unpacks the nibbles with RVV, multiplies
 * each block by its scale in u32 lanes and accumulates. It keeps integer
 * semantics on purpose: the host reference can then check it exactly, which is
 * what a rate is allowed to rest on. Lanes are flushed into a scalar total every
 * 1,024 blocks (a lane holds at most 15*65535 per block) so the u32 cannot wrap.
 */
void read_kernel_rvv_gemv(spert::Context* context, const uint8_t* data, uint64_t bytes,
                          uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t per_tile = bytes / tiles;
    const uint64_t blocks = per_tile / 16;
    const uint8_t* base = data + static_cast<uint64_t>(tile) * per_tile;
    const uint8_t* scales = data + bytes + static_cast<uint64_t>(tile) * blocks * 2;
    uint64_t accumulator = 0;
    const size_t vl = __riscv_vsetvl_e8m1(16);
    vuint32m4_t lanes = __riscv_vmv_v_x_u32m4(0, vl);

    for (uint32_t pass = 0; pass < passes; ++pass) {
        uint64_t block = 0;

        for (; block + 2 <= blocks; block += 2) {
            for (int index = 0; index < 2; ++index) {
                const vuint8m1_t packed =
                    __riscv_vle8_v_u8m1(base + (block + index) * 16, vl);
                const vuint8m1_t low = __riscv_vand_vx_u8m1(packed, 0x0f, vl);
                const vuint8m1_t high = __riscv_vsrl_vx_u8m1(packed, 4, vl);
                uint16_t scale = 0;

                __builtin_memcpy(&scale, scales + (block + index) * 2, sizeof(scale));
                lanes = __riscv_vadd_vv_u32m4(
                    lanes,
                    __riscv_vmul_vx_u32m4(
                        __riscv_vzext_vf4_u32m4(low, vl), (uint32_t)scale, vl),
                    vl);
                lanes = __riscv_vadd_vv_u32m4(
                    lanes,
                    __riscv_vmul_vx_u32m4(
                        __riscv_vzext_vf4_u32m4(high, vl), (uint32_t)scale, vl),
                    vl);
            }
            /* Drain every 64 blocks, not every 1,024: a lane holds at most
             * 15 * 65,535 per block, and the u32 *reduction* sums 16 lanes, so the
             * window that has to fit in u32 is 16 x the per-lane total. 64 blocks
             * keeps that near 1e9. Draining less often overflowed and the host
             * reference caught it. */
            if ((block & 63) == 0) {
                const vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, vl);
                const vuint32m1_t drained =
                    __riscv_vredsum_vs_u32m4_u32m1(lanes, zero, vl);

                accumulator += (uint64_t)__riscv_vmv_x_s_u32m1_u32(drained);
                lanes = __riscv_vmv_v_x_u32m4(0, vl);
            }
        }
        for (; block < blocks; ++block) {
            uint16_t scale = 0;
            uint64_t sum = 0;

            for (uint64_t index = 0; index < 16; ++index) {
                const uint8_t byte = base[block * 16 + index];

                sum += (byte & 0x0fu) + (byte >> 4);
            }
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            accumulator += sum * (uint64_t)scale;
        }
    }
    {
        const vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, vl);
        const vuint32m1_t total = __riscv_vredsum_vs_u32m4_u32m1(lanes, zero, vl);

        accumulator += (uint64_t)__riscv_vmv_x_s_u32m1_u32(total);
    }
    out[tile] = accumulator;
}

/* Same probe with a different operand configuration, to see whether "only two
 * bytes of A participate" is a property of one vsetvli setting or of the cluster.
 * SEW=4 cannot be asked for at all: this toolchain's assembler rejects
 * `vsetvli t0, zero, e4, m1` with "illegal operands", so the smallest MAC unit the
 * specification's table lists is reachable only through raw instruction encoding
 * and is not covered here. */
void probe_kernel_e16(spert::Context* context, const uint8_t* data, uint64_t bytes,
                      uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);

    (void)bytes;
    (void)passes;
    for (int step = 0; step < 2; ++step) {
        const uint32_t tile_case = tile * 2 + (uint32_t)step;
        const uint8_t* slice = data + (uint64_t)tile_case * 32;
        int32_t* acc = reinterpret_cast<int32_t*>(out + (uint64_t)tile_case * 16);

        for (int index = 0; index < 8; ++index)
            acc[index] = 0;
        __asm__ volatile(
            "vsetvli t0, zero, e32, m2   \n\t"
            "vle32.v v28, (%[ACC])       \n\t"
            "vsetvli t0, zero, e16, m1   \n\t"
            "vle16.v v0, (%[A])          \n\t"
            "vle16.v v1, (%[B])          \n\t"
            "vmadot  v28, v0, v1         \n\t"
            "vsetvli t0, zero, e32, m2   \n\t"
            "vse32.v v28, (%[ACC])       \n\t"
            : : [A] "r"(slice), [B] "r"(slice + 16), [ACC] "r"(acc)
            : "cc", "t0", "v0", "v1", "v28", "memory");
        uint64_t mask = 0;

        for (int index = 0; index < 8; ++index)
            if (acc[index] != 0)
                mask |= (uint64_t)1 << index;
        out[tile_case * 16 + 8] = mask | 0x1000000000ull;
    }
}

void probe_kernel_av8(spert::Context* context, const uint8_t* data, uint64_t bytes,
                      uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);

    (void)bytes;
    (void)passes;
    for (int step = 0; step < 2; ++step) {
        const uint32_t tile_case = tile * 2 + (uint32_t)step;
        const uint8_t* slice = data + (uint64_t)tile_case * 32;
        int32_t* acc = reinterpret_cast<int32_t*>(out + (uint64_t)tile_case * 16);

        for (int index = 0; index < 8; ++index)
            acc[index] = 0;
        __asm__ volatile(
            "vsetvli t0, zero, e32, m2   \n\t"
            "vle32.v v28, (%[ACC])       \n\t"
            "vsetivli t0, 8, e8, m1      \n\t"
            "vle8.v  v0, (%[A])          \n\t"
            "vle8.v  v1, (%[B])          \n\t"
            "vmadot  v28, v0, v1         \n\t"
            "vsetvli t0, zero, e32, m2   \n\t"
            "vse32.v v28, (%[ACC])       \n\t"
            : : [A] "r"(slice), [B] "r"(slice + 16), [ACC] "r"(acc)
            : "cc", "t0", "v0", "v1", "v28", "memory");
        uint64_t mask = 0;

        for (int index = 0; index < 8; ++index)
            if (acc[index] != 0)
                mask |= (uint64_t)1 << index;
        out[tile_case * 16 + 8] = mask | 0x1000000000ull;
    }
}

/* Layout probe for vmadot on this cluster.
 *
 * The IME specification pins the operand *shapes* down (Copies*(M,K) in vs1,
 * Copies*(K,N) in vs2, an even vd pair for Copies*(M,N) int32) but its operand
 * figures are images and its integer pseudocode indexes C as an (M,N) matrix with
 * i*K + j, which cannot hold when M != N. A kernel written from that prose alone
 * runs on the A100 and produces wrong answers, and the vendor demo is a VLEN=256
 * kernel whose layout does not transfer to this cluster's VLEN=128.
 *
 * So the permutation is measured here instead of read: the buffer carries A (16
 * bytes) followed by B (16 bytes), the caller puts a single 1 in one position and
 * 1s everywhere else, and the resulting C says which (row, k) that position
 * feeds. Same method this workspace used for tracepoint attachment: source as
 * hypothesis, probe as fact. */
void probe_kernel(spert::Context* context, const uint8_t* data, uint64_t bytes,
                  uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);

    (void)bytes;
    (void)passes;
    /* The accumulator lives in the buffer the launch was given, not on the host
     * stack: every kernel in this repository that works writes only into `out`,
     * and the first probe versions that crashed differed in exactly that way. */
    /* The grid is the granted core count, which is the only shape this runtime
     * has accepted so far (grid 1 spun, grid 32 faulted, grid == cores works in
     * every other kernel here), so each tile walks four of the 32 cases. */
    /* Register assignment follows the specification rather than convenience: with
     * VLEN=128 and LMUL=1 a register is 16 bytes, and the integer form takes A
     * from VS1 *and* VS1+1 while B sits in VS2. The first probe loaded A into v0
     * and B into v1, so the instruction read its second A register out of the B
     * data - which is what made the (p,q) table look sparse and asymmetric. Each
     * case is therefore 64 bytes: A at 0..31, B at 32..47. */
    for (int step = 0; step < 64; ++step) {
        const uint32_t tile_case = tile * 64 + (uint32_t)step;
        const uint8_t* slice = data + (uint64_t)tile_case * 64;
        /* C occupies two sequential vector registers (the specification's even-vd
         * rule), so a case needs 16 int32 - two copies of a 2x2 int32 tile at
         * VLEN=128 - and the mask goes after them, not inside them. The first
         * version left only one register's worth and the second register's values
         * landed on the mask. */
        int32_t* acc = reinterpret_cast<int32_t*>(out + (uint64_t)tile_case * 16);

        for (int index = 0; index < 8; ++index)
            acc[index] = 0;
        __asm__ volatile(
            "vsetvli t0, zero, e32, m2   \n\t"
            "vle32.v v28, (%[ACC])       \n\t"
            "vsetvli t0, zero, e8, m1    \n\t"
            "vle8.v  v0, (%[A])          \n\t"
            "vle8.v  v1, (%[A2])         \n\t"
            "vle8.v  v2, (%[B])          \n\t"
            "vmadot  v28, v0, v2         \n\t"
            "vsetvli t0, zero, e32, m2   \n\t"
            "vse32.v v28, (%[ACC])       \n\t"
            : : [A] "r"(slice), [A2] "r"(slice + 16), [B] "r"(slice + 32),
                [ACC] "r"(acc)
            : "cc", "t0", "v0", "v1", "v2", "v28", "memory");
        {
            uint64_t mask = 0;

            for (int index = 0; index < 8; ++index)
                if (acc[index] != 0)
                    mask |= (uint64_t)1 << index;
            /* keep the mask beside the accumulator so the caller can read both */
            out[tile_case * 16 + 8] = mask;
        }
    }
}

static int ime_layout_probe(spert::Stream& stream, const uint32_t cores)
{
    /* One launch, 32 cases: tiles 0-15 put a single 1 in one A position with B all
     * ones, tiles 16-31 do the same for B. With a single 1 in one operand the
     * product is 0 or 1 per accumulator, so each tile reports which of the eight
     * C slots answered - the whole layout in one result vector. */
    /* Both buffers are page-aligned mappings rather than vector storage: the
     * probe still reports free(): invalid next size at exit after printing its
     * mapping, and a mapping that the runtime cannot corrupt into the heap is
     * worth more than a diagnosis of which allocator boundary it crosses. */
    bool huge = false;
    uint8_t* buffer = allocate(512 * 64, &huge);
    uint8_t* out_bytes = allocate(512 * 16 * sizeof(uint64_t), &huge);
    uint64_t* out = reinterpret_cast<uint64_t*>(out_bytes);

    if (buffer == nullptr || out_bytes == nullptr)
        return 1;
    std::memset(buffer, 0, 512 * 64);
    std::memset(out_bytes, 0, 512 * 16 * sizeof(uint64_t));
    std::fprintf(stderr, "ime-layout: buffers are page-aligned mappings\n");

    /* One 1 in A and one in B per case, over all 16x16 combinations: two positions
     * multiply only if they share the MAC's contraction index, so the nonzero
     * (p,q) pairs partition the positions into k-classes, and the C slot that
     * answers names the (row, column) each position carries. That is the whole
     * packing question, asked in one launch. */
    for (int tile_case = 0; tile_case < 512; ++tile_case) {
        uint8_t* slice = buffer + tile_case * 64;
        const int p = tile_case / 16;      /* A position 0..31 across v0, v1 */
        const int q = tile_case % 16;      /* B position 0..15 inside v2 */

        std::memset(slice, 0, 64);
        slice[p] = 1;
        slice[32 + q] = 1;
    }
    std::fprintf(stderr, "ime-layout: one launch, grid %u (the granted core count)\n",
                 cores);
    std::fprintf(stderr, "ime-layout: A is 32 bytes (v0,v1), B is 16 (v2); 512 cases\n");
    auto future = stream.launch(spert::Grid(cores), probe_kernel, buffer, (uint64_t)8192,
                                1u, out);

    if (!future.valid()) {
        std::fprintf(stderr, "ime-layout: launch refused\n");
        return 1;
    }
    std::fprintf(stderr, "ime-layout: launched, waiting for sync\n");
    future.sync();
    std::fprintf(stderr, "ime-layout: synced; only the cases that answer are printed\n");
    for (int tile = 0; tile < 512; ++tile) {
        const uint64_t mask = out[tile * 16 + 8];
        if (!mask)
            continue;
        const int32_t* acc = reinterpret_cast<const int32_t*>(&out[tile * 16]);
        const int32_t* acc2 = acc + 8;

        std::fprintf(stderr,
                     "  p=%2d q=%2d -> copy0 [%d %d %d %d %d %d %d %d] "
                     "copy1 [%d %d %d %d %d %d %d %d] mask 0x%02llx\n",
                     tile / 16, tile % 16,
                     acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7],
                     acc2[0], acc2[1], acc2[2], acc2[3], acc2[4], acc2[5], acc2[6],
                     acc2[7], (unsigned long long)mask);
    }

    /* The A sweep only, under the configurations the assembler accepts: A has a
     * single 1 at p and B is all ones, so a live p means that byte of A reaches
     * the MAC. */
    {
        const struct {
            const char* name;
            void (*kernel)(spert::Context*, const uint8_t*, uint64_t, uint32_t, uint64_t*);
        } configs[] = {
            {"e8  (this probe)", probe_kernel},
            {"e16", probe_kernel_e16},
            {"e8 avl=8", probe_kernel_av8},
        };

        for (int index = 0; index < 16; ++index) {
            uint8_t* slice = buffer + index * 32;

            std::memset(slice, 0, 32);
            slice[index % 16] = 1;
            for (int other = 16; other < 32; ++other)
                slice[other] = 1;
        }
        for (const auto& config : configs) {
            std::memset(out_bytes, 0, 256 * 16 * sizeof(uint64_t));
            auto attempt = stream.launch(spert::Grid(cores), config.kernel, buffer,
                                         (uint64_t)512, 1u, out);

            if (!attempt.valid()) {
                std::fprintf(stderr, "config %-16s launch refused\n", config.name);
                continue;
            }
            attempt.sync();
            std::fprintf(stderr, "config %-16s live A positions:", config.name);
            for (int index = 0; index < 16; ++index)
                if (out[index * 16 + 8] & 0xff)
                    std::fprintf(stderr, " %d", index);
            std::fprintf(stderr, "\n");
        }
    }
    return 0;
}

/* The same packed read as `read_kernel_packed`, with the nibble extraction done
 * in vector instructions.
 *
 * The scalar version of this loop reads 2.7 GiB/s and the question is how much
 * of that is the arithmetic and how much is the memory: the plain `word` stream
 * reaches 24.7 GiB/s and the vendor's own one-row kernel 9.0 GiB/s, so if a
 * vector unpack lands near 9 the structure can feed a multi-row kernel and if it
 * lands near 3 it cannot.
 *
 * This is the simple form: each 16-byte block is loaded, its nibbles are summed
 * across lanes with one `vredsum`, and the block's scale is added in scalar code
 * - the arithmetic the scalar variant and the host reference perform.
 *
 * A lane-wise version that reduced only once at the end was tried and is not
 * kept: it needs periodic draining to avoid u16 overflow, and the version
 * written here lost its vector contribution entirely (the mismatch equalled one
 * pass of nibble sums), so a kernel nobody has time to chase is worth less than a
 * slower one whose checksum passes. */
void read_kernel_packed_vector(spert::Context* context, const uint8_t* data, uint64_t bytes,
                               uint32_t passes, uint64_t* out) {
    const uint32_t tile = context->program_id(0);
    const uint32_t tiles = context->grid_dim(0);
    const uint64_t per_tile = bytes / tiles;
    const uint64_t blocks = per_tile / 16;
    const uint8_t* base = data + static_cast<uint64_t>(tile) * per_tile;
    const uint8_t* scales = data + bytes + static_cast<uint64_t>(tile) * blocks * 2;
    uint64_t accumulator = 0;
    const size_t vl = __riscv_vsetvl_e8m1(16);

    for (uint32_t pass = 0; pass < passes; ++pass) {
        uint64_t block = 0;

        for (; block + 4 <= blocks; block += 4) {
            for (int index = 0; index < 4; ++index) {
                const vuint8m1_t packed =
                    __riscv_vle8_v_u8m1(base + (block + index) * 16, vl);
                const vuint8m1_t nibbles = __riscv_vadd_vv_u8m1(
                    __riscv_vand_vx_u8m1(packed, 0x0f, vl),
                    __riscv_vsrl_vx_u8m1(packed, 4, vl), vl);
                const vuint16m1_t zero = __riscv_vmv_v_x_u16m1(0, vl);
                const vuint16m1_t wide =
                    __riscv_vredsum_vs_u16m2_u16m1(__riscv_vzext_vf2_u16m2(nibbles, vl), zero,
                                             vl);

                accumulator += (uint64_t)__riscv_vmv_x_s_u16m1_u16(wide);
            }
            for (int index = 0; index < 4; ++index) {
                uint16_t scale = 0;

                __builtin_memcpy(&scale, scales + (block + index) * 2, sizeof(scale));
                accumulator += scale;
            }
        }
        for (; block < blocks; ++block) {
            uint16_t scale = 0;
            uint64_t nibbles = 0;

            for (uint64_t index = 0; index < 16; ++index) {
                const uint8_t byte = base[block * 16 + index];

                nibbles += (byte & 0x0fu) + (byte >> 4);
            }
            __builtin_memcpy(&scale, scales + block * 2, sizeof(scale));
            accumulator += nibbles + scale;
        }
    }
    out[tile] = accumulator;
}
#endif

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
    /* Unbuffered stderr: a probe that faults must still leave its last steps
     * visible, which a block-buffered stdout does not. */
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    bool layout_probe = false;
    bool sfq_probe = false;
    bool smthp_probe = false;
    bool smthp_control = false;

    if (argc == 2 && (std::strcmp(argv[1], "-h") == 0 ||
                      std::strcmp(argv[1], "--help") == 0)) {
        usage();
        return 0;
    }
    for (int index = 1; index < argc; ++index) {
        long value = std::strtol(argv[index], nullptr, 10);

        if (std::strcmp(argv[index], "--ime-layout") == 0) {
            layout_probe = true;
            continue;
        }
        if (std::strcmp(argv[index], "--sfqmac") == 0) {
            sfq_probe = true;
            continue;
        }
        if (std::strcmp(argv[index], "--smthp") == 0) {
            smthp_probe = true;
            continue;
        }
        if (std::strcmp(argv[index], "--smthpctl") == 0) {
            smthp_probe = true;
            smthp_control = true;
            continue;
        }
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
#ifdef HAVE_RVV
    if (layout_probe)
        return ime_layout_probe(stream, cores);
    if (sfq_probe) {
        /* One instruction on every core, then read what came back. A trap shows
         * up as a failed launch; a value means the silicon implements it. */
        bool huge = false;
        uint8_t* inputs = allocate(32, &huge);
        uint8_t* results = allocate(1024 + cores * 32 * sizeof(uint64_t), &huge);

        if (inputs == nullptr || results == nullptr)
            return 1;
        std::memset(inputs, 1, 32);
        std::memset(results, 0, 1024 + cores * 32 * sizeof(uint64_t));
        std::fprintf(stderr, "sfqmac: launching one sf.vqmacc.4x8x4 per core\n");
        auto attempt = stream.launch(spert::Grid(cores), probe_sfq_kernel, inputs, (uint64_t)32,
                                     1u, reinterpret_cast<uint64_t*>(results));

        if (!attempt.valid()) {
            std::fprintf(stderr, "sfqmac: launch refused\n");
            return 1;
        }
        attempt.sync();
        std::fprintf(stderr, "sfqmac: synced\n");
        for (uint32_t tile = 0; tile < cores; ++tile) {
            const int32_t* acc = reinterpret_cast<const int32_t*>(results + tile * 128);

            std::fprintf(stderr, "  core %u: [%d %d %d %d %d %d %d %d]\n", tile,
                         acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
        }
        return 0;
    }
    if (smthp_probe) {
        /* One `smt.vmadotsu.hp` per core. The inputs are uniform, so the expected
         * answer does not depend on the tile extent: 32 products of 1x1, scaled by
         * FP16 1.0, is 32.0 = 0x5000 in every FP16 lane. */
        bool huge = false;
        uint8_t* inputs = allocate(32, &huge);
        uint8_t* results = allocate(cores * 16 * sizeof(uint64_t), &huge);

        if (inputs == nullptr || results == nullptr)
            return 1;
        for (int index = 0; index < 8; ++index) { /* scale: FP16 1.0, all lanes */
            inputs[index * 2 + 0] = 0x00;
            inputs[index * 2 + 1] = 0x3c;
        }
        for (int index = 0; index < 8; ++index) { /* operands: every nibble is 1 */
            inputs[16 + index * 2 + 0] = 0x11;
            inputs[16 + index * 2 + 1] = 0x11;
        }
        std::memset(results, 0, cores * 16 * sizeof(uint64_t));
        std::fprintf(stderr, "smthp: launching %s per core\n",
                     smthp_control ? "the control kernel (no matrix instruction)"
                                   : "one smt.vmadotsu.hp");
        auto attempt = stream.launch(spert::Grid(cores),
                                     smthp_control ? probe_smthp_control_kernel
                                                   : probe_smthp_kernel, inputs,
                                     (uint64_t)32, 1u,
                                     reinterpret_cast<uint64_t*>(results));

        if (!attempt.valid()) {
            std::fprintf(stderr, "smthp: launch refused\n");
            return 1;
        }
        attempt.sync();
        std::fprintf(stderr, "smthp: synced\n");
        {
            const uint64_t* flat = reinterpret_cast<const uint64_t*>(results);

            std::fprintf(stderr, "  out[0..7] =");
            for (uint32_t tile = 0; tile < cores; ++tile)
                std::fprintf(stderr, " 0x%llx",
                             static_cast<unsigned long long>(flat[tile]));
            std::fprintf(stderr, "\n");
        }
        for (uint32_t tile = 0; tile < cores; ++tile) {
            const uint16_t* lanes =
                reinterpret_cast<const uint16_t*>(results + 1024 + tile * 256);
            const uint64_t scalar =
                reinterpret_cast<const uint64_t*>(results + tile * 256)[24];

            std::fprintf(stderr, "  core %u: v24", tile);
            for (int index = 0; index < 8; ++index)
                std::fprintf(stderr, " 0x%04x", lanes[index]);
            std::fprintf(stderr, "  v25");
            for (int index = 0; index < 8; ++index)
                std::fprintf(stderr, " 0x%04x", lanes[8 + index]);
            std::fprintf(stderr, "  tile-id");
            for (int index = 0; index < 8; ++index)
                std::fprintf(stderr, " 0x%04x", lanes[16 + index]);
            std::fprintf(stderr, "  scalar=0x%llx\n",
                         static_cast<unsigned long long>(scalar));
        }
        return 0;
    }
#endif

    using kernel_fn = void (*)(spert::Context*, const uint8_t*, uint64_t, uint32_t,
                               uint64_t*);
    enum class kind { bytes, words, packed, packed_scaled, tiled, gemv1, gemv4, gemv8 };
    const struct {
        const char* name;
        kernel_fn kernel;
        kind how;
    } variants[] = {
        {"byte", read_kernel, kind::bytes},
        {"word", read_kernel_words, kind::words},
        {"pack4", read_kernel_packed, kind::packed},
#ifdef HAVE_RVV
        {"pack4v", read_kernel_packed_vector, kind::packed},
        {"rvvgemv", read_kernel_rvv_gemv, kind::packed_scaled},
#endif
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
            } else if (variant.how == kind::packed_scaled) {
                expected = host_checksum_packed_scaled(buffer, bytes, passes);
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
                           (variant.how == kind::packed || variant.how == kind::packed_scaled ||
                            variant.how == kind::gemv1 ||
                            variant.how == kind::gemv4 || variant.how == kind::gemv8
                                ? static_cast<double>(scale_bytes)
                                : 0.0);
            if (variant.how == kind::tiled)
                moved = static_cast<double>((bytes / kTileBytes) * kTileBytes);
            double mib_per_s = seconds > 0.0 ? moved * passes / kMiB / seconds : 0.0;
            std::printf("%10zu %6s %8u %8u %10.4f %12.1f %s\n", size_mib,
                        variant.name, cores, passes, seconds, mib_per_s,
                        got == expected ? "ok" : "MISMATCH");
            if (got != expected)
                std::printf("           %s: expected %llu got %llu (delta %lld)\n",
                            variant.name, (unsigned long long)expected,
                            (unsigned long long)got,
                            (long long)got - (long long)expected);
            std::printf("RESULT size_mib=%zu kernel=%s cores=%u passes=%u seconds=%.4f mib_per_s=%.1f moved_mib=%.1f hugepages=%d checksum=%s\n",
                        size_mib, variant.name, cores, passes, seconds, mib_per_s,
                        moved / kMiB, huge ? 1 : 0,
                        got == expected ? "ok" : "mismatch");
        }
        munmap(buffer, bytes + scale_bytes);
    }
    return 0;
}
